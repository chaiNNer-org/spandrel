# type: ignore  # noqa: PGH003
import itertools
import math
from collections.abc import Iterable
from typing import TypeVar

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn.modules.utils import _pair

"""
28-Sep-21
https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/df156f569a4b0c658209fb67244629b879861034/model/conv/DynamicConv.py#L40
"""



T = TypeVar("T", bound=nn.Module)
def normal_init(module, mean=0, std=1.0, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    def __init__(self, in_channels: int, out_ch: int, scale: int = 2, groups: int = 4):
        super().__init__()

        assert in_channels >= groups and in_channels % groups == 0
        out_channels = 2 * groups * scale ** 2

        self.scale = scale
        self.groups = groups
        self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.register_buffer('init_pos', self._init_pos())

        normal_init(self.end_conv, std=0.001)
        normal_init(self.offset, std=0.001)
        constant_init(self.scope, val=0.)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (torch.stack(torch.meshgrid([h, h], indexing="ij")).transpose(1, 2)
                .repeat(1, self.groups, 1).reshape(1, -1, 1, 1))

    def forward(self, x):
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij")
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return self.end_conv(F.grid_sample(x.reshape(B * self.groups, -1, H, W),
                                            coords, mode='bilinear',
                                            align_corners=False,
                                            padding_mode="border")
                              .view(B, -1, self.scale * H, self.scale * W))


class Conv2dWrapper(nn.Conv2d):
    """
    Wrapper for pytorch Conv2d class which can take additional parameters(like temperature) and ignores them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


class TempModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, temperature) -> torch.Tensor:
        return x


class BaseModel(TempModule):
    def __init__(self, ConvLayer):
        super().__init__()
        self.ConvLayer = ConvLayer


class TemperatureScheduler:
    def __init__(self, initial_value, final_value=None, final_epoch=None):
        self.initial_value = initial_value
        self.final_value = final_value if final_value else initial_value
        self.final_epoch = final_epoch if final_epoch else 1
        self.step = (
            0
            if self.final_epoch == 1
            else (final_value - initial_value) / (final_epoch - 1)
        )

    def get(self, crt_epoch=None):
        crt_epoch = crt_epoch if crt_epoch else self.final_epoch
        return self.initial_value + (min(crt_epoch, self.final_epoch) - 1) * self.step


class CustomSequential(TempModule):
    """Sequential container that supports passing temperature to TempModule"""

    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList(args)

    def forward(self, x, temperature):
        for layer in self.layers:
            if isinstance(layer, TempModule):
                x = layer(x, temperature)
            else:
                x = layer(x)
        return x

    def __getitem__(self, idx):
        return CustomSequential(*list(self.layers[idx]))
        # if isinstance(idx, slice):
        #     return self.__class__(OrderedDict(list(self.layers.items())[idx]))
        # else:
        #     return self._get_item_by_idx(self.layers.values(), idx)


# Implementation inspired from
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py#L38 and
# https://github.com/pytorch/pytorch/issues/7455
class SmoothNLLLoss(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super().__init__()
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, prediction, target):
        with torch.no_grad():
            smooth_target = torch.zeros_like(prediction)
            n_class = prediction.size(self.dim)
            smooth_target.fill_(self.smoothing / (n_class - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-smooth_target * prediction, dim=self.dim))


class Attention(nn.Module):
    def __init__(self,in_planes,ratio,K,temprature=30,init_weight=True):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.temprature=temprature
        assert in_planes>ratio
        hidden_planes=in_planes//ratio
        self.net=nn.Sequential(
            nn.Conv2d(in_planes,hidden_planes,kernel_size=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes,K,kernel_size=1,bias=False)
        )

        if(init_weight):
            self._initialize_weights()

    def update_temprature(self):
        if(self.temprature>1):
            self.temprature-=1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        att=self.avgpool(x) #bs,dim,1,1
        att=self.net(att).view(x.shape[0],-1) #bs,K
        return F.softmax(att/self.temprature,-1)


class DynamicConvolution(TempModule):
    def __init__(self,K,grounps,in_planes,out_planes,kernel_size,stride=1,padding=0,dilation=1,ratio=2,bias=True,temprature=30,init_weight=True):
        super().__init__()
        # k = number of kernels
        self.in_planes=in_planes
        self.out_planes=out_planes
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=grounps
        self.bias=bias
        self.K=K
        self.init_weight=init_weight
        self.attention=Attention(in_planes=in_planes,ratio=ratio,K=K,temprature=temprature,init_weight=init_weight)

        self.weight=nn.Parameter(torch.randn(K,out_planes,in_planes//grounps,kernel_size,kernel_size),requires_grad=True)
        if(bias):
            self.bias=nn.Parameter(torch.randn(K,out_planes),requires_grad=True)
        else:
            self.bias=None
        if(self.init_weight):
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self,x):
        bs,in_planels,h,w=x.shape
        softmax_att=self.attention(x) #bs,K
        x=x.view(1,-1,h,w)
        weight=self.weight.view(self.K,-1) #K,-1
        aggregate_weight=torch.mm(softmax_att,weight).view(bs*self.out_planes,self.in_planes//self.groups,self.kernel_size,self.kernel_size) #bs*out_p,in_p,k,k

        if(self.bias is not None):
            bias=self.bias.view(self.K,-1) #K,out_p
            aggregate_bias=torch.mm(softmax_att,bias).view(-1) #bs,out_p
            output=F.conv2d(x,weight=aggregate_weight,bias=aggregate_bias,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)
        else:
            output=F.conv2d(x,weight=aggregate_weight,bias=None,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)

        output=output.view(bs,self.out_planes,h,w)
        return output

class FlexibleKernelsDynamicConvolution:
    def __init__(self, Base, nof_kernels, reduce):
        if isinstance(nof_kernels, Iterable):
            self.nof_kernels_it = iter(nof_kernels)
        else:
            self.nof_kernels_it = itertools.cycle([nof_kernels])
        self.Base = Base
        self.reduce = reduce

    def __call__(self, *args, **kwargs):
        return self.Base(next(self.nof_kernels_it), self.reduce, *args, **kwargs)


def dynamic_convolution_generator(nof_kernels, reduce):
    return FlexibleKernelsDynamicConvolution(DynamicConvolution, nof_kernels, reduce)


class Conv3XC(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, gain: int = 1, s: int = 1, bias: bool = True
    ):
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )
        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )
        if self.training:
            trunc_normal_(self.sk.weight, std=0.02)
            trunc_normal_(self.eval_conv.weight, std=0.02)

        if self.training is False:
            self.eval_conv.weight.requires_grad = False
            self.eval_conv.bias.requires_grad = False
            self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]
        )

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat

    def forward(self, x):
        if self.training:
            x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        return out


class SPAB(nn.Module):
    def __init__(self, in_channels: int, end: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.c2_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.c3_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.act1 = nn.Mish(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.end = end

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = self.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        if self.end:
            return out, out1
        return out


class SPABS(nn.Module):
    def __init__(self, feature_channels: int, n_blocks: int = 4, drop: float = 0.0):
        super().__init__()
        self.block_1 = SPAB(feature_channels)

        self.block_n = nn.Sequential(*[SPAB(feature_channels) for _ in range(n_blocks)])
        self.block_end = SPAB(feature_channels, True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain=2, s=1)
        self.conv_cat = nn.Conv2d(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.dropout = nn.Dropout2d(drop)
        if self.training:
            trunc_normal_(self.conv_cat.weight, std=0.02)

    def forward(self, x):
        out_b1 = self.block_1(x)
        out_x = self.block_n(out_b1)
        out_end, out_x_2 = self.block_end(out_x)
        out_end = self.dropout(self.conv_2(out_end))
        return self.conv_cat(torch.cat([x, out_end, out_b1, out_x_2], 1))


class sudo_SPANPlus(nn.Module):
    """Modified from 'Swift Parameter-free Attention Network for Efficient Super-Resolution':
    https://arxiv.org/abs/2311.12770
    """

    def __init__(
        self,
        num_in_ch: int = 12,
        num_out_ch: int = 3,
        blocks: list = [4],
        feature_channels: int = 64,
        upscale: int = 2,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.upscale = upscale
        self.shrink = nn.PixelUnshuffle(upscale)
        in_channels = num_in_ch
        if not isinstance(blocks, list):
            blocks = [int(blocks)]
        if not self.training:
            drop_rate = 0
        self.feats = nn.Sequential(
            *[Conv3XC(in_channels, feature_channels, gain=2, s=1)]
            + [SPABS(feature_channels, n_blocks, drop_rate) for n_blocks in blocks]
        )
        self.dynamic_prio = DynamicConvolution(
            3,
            1,
            in_planes=feature_channels,
            out_planes=feature_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.upsampler = DySample(feature_channels, feature_channels, self.upscale*2)
        self.dynamic = DynamicConvolution(
            3,
            1,
            in_planes=feature_channels,
            out_planes=num_out_ch,
            kernel_size=3,
            padding=1,
            bias=True,
        )


    def forward(self, x):
        n, c, h, w = x.shape
        ph = ((h - 1) // 8 + 1) * 8
        pw = ((w - 1) // 8 + 1) * 8
        padding = (0, pw - w, 0, ph - h)
        x = F.pad(x, padding)

        x = self.shrink(x)
        out = self.feats(x)
        out = self.dynamic_prio(out)
        out = self.upsampler(out)
        out = self.dynamic(out)
        return out[:, :, : h * self.upscale, : w * self.upscale]
