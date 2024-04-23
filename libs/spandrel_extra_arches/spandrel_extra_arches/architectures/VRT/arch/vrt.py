# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
from distutils.version import LooseVersion
from functools import lru_cache, reduce
from operator import mul
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.modules.utils import _pair, _single  # type: ignore

from spandrel.util import store_hyperparameters
from spandrel.util.timm import DropPath, trunc_normal_


class ModulatedDeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(  # type: ignore
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))  # type: ignore
        else:
            self.register_parameter("bias", None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:  # type: ignore
            self.bias.data.zero_()

    # def forward(self, x, offset, mask):
    #     return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
    #                                  self.groups, self.deformable_groups)


class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True,
        )
        self.init_weights()

    def init_weights(self):
        super().init_weights()
        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()  # type: ignore


def flow_warp(
    x,
    flow,
    interp_mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
        use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel dimension.
            The mask is generated according to the grid_sample results of the padded dimension.


    Returns:
        Tensor: Warped image or feature map.
    """
    # assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
    _n, _, h, w = x.size()
    # create mesh grid
    # grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)) # an illegal memory access on TITAN RTX + PyTorch1.9.1
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, dtype=x.dtype, device=x.device),
        torch.arange(0, w, dtype=x.dtype, device=x.device),
    )
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # if use_pad_mask: # for PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # scale grid to [-1,1]
    if (
        interp_mode == "nearest4"
    ):  # todo: bug, no gradient for flow model in this case!!! but the result is good
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(
            x,
            torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3),
            mode="nearest",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        output01 = F.grid_sample(
            x,
            torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3),
            mode="nearest",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        output10 = F.grid_sample(
            x,
            torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3),
            mode="nearest",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        output11 = F.grid_sample(
            x,
            torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3),
            mode="nearest",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        return torch.cat([output00, output01, output10, output11], 1)

    else:
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(
            x,
            vgrid_scaled,
            mode=interp_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        # if use_pad_mask: # for PWCNet
        #     output = _flow_warp_masking(output)

        # TODO, what if align_corners=False
        return output


class DCNv2PackFlowGuided(ModulatedDeformConvPack):
    """Flow-guided deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
        pa_frames (int): The number of parallel warping frames. Default: 2.

    Ref:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 10)
        self.pa_frames = kwargs.pop("pa_frames", 2)

        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                (1 + self.pa_frames // 2) * self.in_channels + self.pa_frames,
                self.out_channels,
                3,
                1,
                1,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 3 * 9 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, "conv_offset"):
            self.conv_offset[-1].weight.data.zero_()  # type: ignore
            self.conv_offset[-1].bias.data.zero_()  # type: ignore

    def forward(self, x, x_flow_warpeds, x_current, flows):
        out = self.conv_offset(torch.cat(x_flow_warpeds + [x_current] + flows, dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        if self.pa_frames == 2:
            offset = offset + flows[0].flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        elif self.pa_frames == 4:
            offset1, offset2 = torch.chunk(offset, 2, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2], dim=1)
        elif self.pa_frames == 6:
            offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
            offset1, offset2, offset3 = torch.chunk(offset, 3, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset3 = offset3 + flows[2].flip(1).repeat(1, offset3.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2, offset3], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )


class BasicModule(nn.Module):
    """Basic Module for SpyNet."""

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(
                in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3
            ),
        )

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, return_levels: list[int] = [5]):
        super().__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])

        self.register_buffer(
            "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(
                0,
                F.avg_pool2d(
                    input=ref[0], kernel_size=2, stride=2, count_include_pad=False
                ),
            )
            supp.insert(
                0,
                F.avg_pool2d(
                    input=supp[0], kernel_size=2, stride=2, count_include_pad=False
                ),
            )

        flow = ref[0].new_zeros(
            [
                ref[0].size(0),
                2,
                int(math.floor(ref[0].size(2) / 2.0)),
                int(math.floor(ref[0].size(3) / 2.0)),
            ]
        )

        for level in range(len(ref)):
            upsampled_flow = (
                F.interpolate(
                    input=flow, scale_factor=2, mode="bilinear", align_corners=True
                )
                * 2.0
            )

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(
                    input=upsampled_flow, pad=[0, 0, 0, 1], mode="replicate"
                )
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(
                    input=upsampled_flow, pad=[0, 1, 0, 0], mode="replicate"
                )

            flow = (
                self.basic_module[level](
                    torch.cat(
                        [
                            ref[level],
                            flow_warp(
                                supp[level],
                                upsampled_flow.permute(0, 2, 3, 1),
                                interp_mode="bilinear",
                                padding_mode="border",
                            ),
                            upsampled_flow,
                        ],
                        1,
                    )
                )
                + upsampled_flow
            )

            if level in self.return_levels:
                scale = (
                    2 ** (5 - level)
                )  # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                flow_out = F.interpolate(
                    input=flow,
                    size=(h // scale, w // scale),
                    mode="bilinear",
                    align_corners=False,
                )
                flow_out[:, 0, :, :] *= float(w // scale) / float(w_floor // scale)
                flow_out[:, 1, :, :] *= float(h // scale) / float(h_floor // scale)
                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(
            input=ref, size=(h_floor, w_floor), mode="bilinear", align_corners=False
        )
        supp = F.interpolate(
            input=supp, size=(h_floor, w_floor), mode="bilinear", align_corners=False
        )

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list


def window_partition(x: torch.Tensor, window_size: Sequence[int]):
    """Partition the input into windows. Attention will be conducted within the windows.

    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, reduce(mul, window_size), C)
    )

    return windows


def window_reverse(
    windows: torch.Tensor, window_size: Sequence[int], B: int, D: int, H: int, W: int
):
    """Reverse windows back to the original input. Attention was conducted within the windows.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Get the window size and the shift size"""

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0  # type: ignore

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)  # type: ignore


@lru_cache
def compute_mask(
    D: int,
    H: int,
    W: int,
    window_size: Sequence[int],
    shift_size: Sequence[int],
    device: torch.device,
):
    """Compute attnetion mask for input of size (D, H, W). @lru_cache caches each stage results."""

    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0, 0.0
    )

    return attn_mask


class Upsample(nn.Sequential):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale: int, num_feat: int):
        assert LooseVersion(torch.__version__) >= LooseVersion(
            "1.8.1"
        ), "PyTorch version >= 1.8.1 to support 5D PixelShuffle."

        class Transpose_Dim12(nn.Module):
            """Transpose Dim1 and Dim2 of a tensor."""

            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(
                    nn.Conv3d(
                        num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)
                    )
                )
                m.append(Transpose_Dim12())
                m.append(nn.PixelShuffle(2))
                m.append(Transpose_Dim12())
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(
                nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            )
        elif scale == 3:
            m.append(
                nn.Conv3d(
                    num_feat, 9 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)
                )
            )
            m.append(Transpose_Dim12())
            m.append(nn.PixelShuffle(3))
            m.append(Transpose_Dim12())
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(
                nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            )
        else:
            raise ValueError(
                f"scale {scale} is not supported. " "Supported scales: 2^n and 3."
            )
        super().__init__(*m)


class Mlp_GEGLU(nn.Module):
    """Multilayer perceptron with gated linear unit (GEGLU). Ref. "GLU Variants Improve Transformer".

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


class WindowAttention(nn.Module):
    """Window based multi-head mutual attention and self attention.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        mut_attn (bool): If True, add mutual attention to the module. Default: True
    """

    def __init__(
        self,
        dim: int,
        window_size: Sequence[int],
        num_heads: int,
        qkv_bias=False,
        qk_scale: float | None = None,
        mut_attn=True,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.mut_attn = mut_attn

        # self attention with relative position bias
        self.relative_position_bias_table = nn.Parameter(  # type: ignore
            torch.zeros(
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1),
                num_heads,
            )
        )  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.register_buffer(
            "relative_position_index", self.get_position_index(window_size)
        )
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # mutual attention with sine position encoding
        if self.mut_attn:
            self.register_buffer(
                "position_bias",
                self.get_sine_position_encoding(
                    window_size[1:], dim // 2, normalize=True
                ),
            )
            self.qkv_mut = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(2 * dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask=None):
        """Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """

        # self attention
        B_, N, C = x.shape
        qkv = (
            self.qkv_self(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        x_out = self.attention(
            q, k, v, mask, (B_, N, C), relative_position_encoding=True
        )

        # mutual attention
        if self.mut_attn:
            qkv = (
                self.qkv_mut(x + self.position_bias.repeat(1, 2, 1))  # type: ignore
                .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            (q1, q2), (k1, k2), (v1, v2) = (
                torch.chunk(qkv[0], 2, dim=2),
                torch.chunk(qkv[1], 2, dim=2),
                torch.chunk(qkv[2], 2, dim=2),
            )  # B_, nH, N/2, C
            x1_aligned = self.attention(
                q2, k1, v1, mask, (B_, N // 2, C), relative_position_encoding=False
            )
            x2_aligned = self.attention(
                q1, k2, v2, mask, (B_, N // 2, C), relative_position_encoding=False
            )
            x_out = torch.cat([torch.cat([x1_aligned, x2_aligned], 1), x_out], 2)

        # projection
        x = self.proj(x_out)

        return x

    def attention(self, q, k, v, mask, x_shape, relative_position_encoding=True):
        B_, N, C = x_shape
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:N, :N].reshape(-1)  # type: ignore
            ].reshape(N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww,nH
            attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(
                0
            )  # B_, nH, N, N

        if mask is None:
            attn = self.softmax(attn)
        else:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[
                :, :N, :N
            ].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x

    def get_position_index(self, window_size: Sequence[int]):
        """Get pair-wise relative position index for each token inside the window."""

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w)
        )  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        return relative_position_index

    def get_sine_position_encoding(
        self, HW, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        """Get sine position encoding"""

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        not_mask = torch.ones([1, HW[0], HW[1]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        # BxCxHxW
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()


class TMSA(nn.Module):
    """Temporal Mutual Self Attention (TMSA).

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int] = (6, 8, 8),
        shift_size: Sequence[int] = (0, 0, 0),
        mut_attn=True,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_checkpoint_attn=False,
        use_checkpoint_ffn=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn

        assert (
            0 <= self.shift_size[0] < self.window_size[0]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[1] < self.window_size[1]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[2] < self.window_size[2]
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mut_attn=mut_attn,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp_GEGLU(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer
        )

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size(
            (D, H, W), self.window_size, self.shift_size
        )

        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(
            x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode="constant"
        )

        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # attention / shifted attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(
            attn_windows, window_size, B, Dp, Hp, Wp
        )  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        x = self.drop_path(x)

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        # attention
        if self.use_checkpoint_attn:
            x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = x + self.forward_part1(x, mask_matrix)

        # feed-forward
        if self.use_checkpoint_ffn:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class TMSAG(nn.Module):
    """Temporal Mutual Self Attention Group (TMSAG).

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (6,8,8).
        shift_size (tuple[int]): Shift size for mutual and self attention. Default: None.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int] = [6, 8, 8],
        shift_size: Sequence[int] | None = None,
        mut_attn=True,
        mlp_ratio=2.0,
        qkv_bias=False,
        qk_scale=None,
        drop_path: float | list[float] = 0.0,
        norm_layer=nn.LayerNorm,
        use_checkpoint_attn=False,
        use_checkpoint_ffn=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (
            [i // 2 for i in window_size] if shift_size is None else shift_size
        )

        # build blocks
        self.blocks = nn.ModuleList(
            [
                TMSA(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
                    mut_attn=mut_attn,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint_attn=use_checkpoint_attn,
                    use_checkpoint_ffn=use_checkpoint_ffn,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for attention
        B, _C, D, H, W = x.shape
        window_size, shift_size = get_window_size(
            (D, H, W), self.window_size, self.shift_size
        )
        x = rearrange(x, "b c d h w -> b d h w c")
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)
        x = rearrange(x, "b d h w c -> b c d h w")

        return x


class RTMSA(nn.Module):
    """Residual Temporal Mutual Self Attention (RTMSA). Only used in stage 8.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path: float | list[float] = 0.0,
        norm_layer=nn.LayerNorm,
        use_checkpoint_attn=False,
        use_checkpoint_ffn=False,
    ):
        super().__init__()

        self.residual_group = TMSAG(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mut_attn=False,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint_attn=use_checkpoint_attn,
            use_checkpoint_ffn=use_checkpoint_ffn,
        )

        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)


class Stage(nn.Module):
    """Residual Temporal Mutual Self Attention Group and Parallel Warping.

    Args:
        in_dim (int): Number of input channels.
        dim (int): Number of channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        reshape (str): Downscale (down), upscale (up) or keep the size (none).
        max_residue_magnitude (float): Maximum magnitude of the residual of optical flow.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(
        self,
        in_dim: int,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        mul_attn_ratio=0.75,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path: float | list[float] = 0.0,
        norm_layer=nn.LayerNorm,
        pa_frames=2,
        deformable_groups=16,
        reshape=None,
        max_residue_magnitude=10.0,
        use_checkpoint_attn=False,
        use_checkpoint_ffn=False,
    ):
        super().__init__()
        self.pa_frames = pa_frames

        # reshape the tensor
        if reshape == "none":
            self.reshape = nn.Sequential(
                Rearrange("n c d h w -> n d h w c"),
                nn.LayerNorm(dim),
                Rearrange("n d h w c -> n c d h w"),
            )
        elif reshape == "down":
            self.reshape = nn.Sequential(
                Rearrange(
                    "n c d (h neih) (w neiw) -> n d h w (neiw neih c)", neih=2, neiw=2
                ),
                nn.LayerNorm(4 * in_dim),
                nn.Linear(4 * in_dim, dim),
                Rearrange("n d h w c -> n c d h w"),
            )
        elif reshape == "up":
            self.reshape = nn.Sequential(
                Rearrange(
                    "n (neiw neih c) d h w -> n d (h neih) (w neiw) c", neih=2, neiw=2
                ),
                nn.LayerNorm(in_dim // 4),
                nn.Linear(in_dim // 4, dim),
                Rearrange("n d h w c -> n c d h w"),
            )

        # mutual and self attention
        self.residual_group1 = TMSAG(
            dim=dim,
            depth=int(depth * mul_attn_ratio),
            num_heads=num_heads,
            window_size=(2, window_size[1], window_size[2]),
            mut_attn=True,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint_attn=use_checkpoint_attn,
            use_checkpoint_ffn=use_checkpoint_ffn,
        )
        self.linear1 = nn.Linear(dim, dim)

        # only self attention
        self.residual_group2 = TMSAG(
            dim=dim,
            depth=depth - int(depth * mul_attn_ratio),
            num_heads=num_heads,
            window_size=window_size,
            mut_attn=False,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint_attn=True,
            use_checkpoint_ffn=use_checkpoint_ffn,
        )
        self.linear2 = nn.Linear(dim, dim)

        # parallel warping
        if self.pa_frames:
            self.pa_deform = DCNv2PackFlowGuided(
                dim,
                dim,
                3,
                padding=1,
                deformable_groups=deformable_groups,
                max_residue_magnitude=max_residue_magnitude,
                pa_frames=pa_frames,
            )
            self.pa_fuse = Mlp_GEGLU(dim * (1 + 2), dim * (1 + 2), dim)

    def forward(self, x, flows_backward, flows_forward):
        x = self.reshape(x)
        x = self.linear1(self.residual_group1(x).transpose(1, 4)).transpose(1, 4) + x
        x = self.linear2(self.residual_group2(x).transpose(1, 4)).transpose(1, 4) + x

        if self.pa_frames:
            x = x.transpose(1, 2)
            x_backward, x_forward = getattr(
                self, f"get_aligned_feature_{self.pa_frames}frames"
            )(x, flows_backward, flows_forward)
            x = self.pa_fuse(
                torch.cat([x, x_backward, x_forward], 2).permute(0, 1, 3, 4, 2)
            ).permute(0, 4, 1, 2, 3)

        return x

    def get_aligned_feature_2frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 2 frames."""

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[0][:, i - 1, ...]
            x_i_warped = flow_warp(
                x_i, flow.permute(0, 2, 3, 1), "bilinear"
            )  # frame i+1 aligned towards i
            x_backward.insert(
                0, self.pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow])
            )

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[0][:, i, ...]
            x_i_warped = flow_warp(
                x_i, flow.permute(0, 2, 3, 1), "bilinear"
            )  # frame i-1 aligned towards i
            x_forward.append(
                self.pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow])
            )

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_4frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 4 frames."""

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n, 1, -1):
            x_i = x[:, i - 1, ...]
            flow1 = flows_backward[0][:, i - 2, ...]
            if i == n:
                x_ii = torch.zeros_like(x[:, n - 2, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, n - 3, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_backward[1][:, i - 2, ...]

            x_i_warped = flow_warp(
                x_i, flow1.permute(0, 2, 3, 1), "bilinear"
            )  # frame i+1 aligned towards i
            x_ii_warped = flow_warp(
                x_ii, flow2.permute(0, 2, 3, 1), "bilinear"
            )  # frame i+2 aligned towards i
            x_backward.insert(
                0,
                self.pa_deform(
                    torch.cat([x_i, x_ii], 1),
                    [x_i_warped, x_ii_warped],
                    x[:, i - 2, ...],
                    [flow1, flow2],
                ),
            )

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(-1, n - 2):
            x_i = x[:, i + 1, ...]
            flow1 = flows_forward[0][:, i + 1, ...]
            if i == -1:
                x_ii = torch.zeros_like(x[:, 1, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_forward[1][:, i, ...]

            x_i_warped = flow_warp(
                x_i, flow1.permute(0, 2, 3, 1), "bilinear"
            )  # frame i-1 aligned towards i
            x_ii_warped = flow_warp(
                x_ii, flow2.permute(0, 2, 3, 1), "bilinear"
            )  # frame i-2 aligned towards i
            x_forward.append(
                self.pa_deform(
                    torch.cat([x_i, x_ii], 1),
                    [x_i_warped, x_ii_warped],
                    x[:, i + 2, ...],
                    [flow1, flow2],
                )
            )

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_6frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 6 frames."""

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n + 1, 2, -1):
            x_i = x[:, i - 2, ...]
            flow1 = flows_backward[0][:, i - 3, ...]
            if i == n + 1:
                x_ii = torch.zeros_like(x[:, -1, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, -1, ...])
                x_iii = torch.zeros_like(x[:, -1, ...])
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
            elif i == n:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = torch.zeros_like(x[:, -1, ...])
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = x[:, i, ...]
                flow3 = flows_backward[2][:, i - 3, ...]

            x_i_warped = flow_warp(
                x_i, flow1.permute(0, 2, 3, 1), "bilinear"
            )  # frame i+1 aligned towards i
            x_ii_warped = flow_warp(
                x_ii, flow2.permute(0, 2, 3, 1), "bilinear"
            )  # frame i+2 aligned towards i
            x_iii_warped = flow_warp(
                x_iii, flow3.permute(0, 2, 3, 1), "bilinear"
            )  # frame i+3 aligned towards i
            x_backward.insert(
                0,
                self.pa_deform(
                    torch.cat([x_i, x_ii, x_iii], 1),
                    [x_i_warped, x_ii_warped, x_iii_warped],
                    x[:, i - 3, ...],
                    [flow1, flow2, flow3],
                ),
            )

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow1 = flows_forward[0][:, i, ...]
            if i == 0:
                x_ii = torch.zeros_like(x[:, 0, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
                x_iii = torch.zeros_like(x[:, 0, ...])
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
            elif i == 1:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = torch.zeros_like(x[:, 0, ...])
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = x[:, i - 2, ...]
                flow3 = flows_forward[2][:, i - 2, ...]

            x_i_warped = flow_warp(
                x_i, flow1.permute(0, 2, 3, 1), "bilinear"
            )  # frame i-1 aligned towards i
            x_ii_warped = flow_warp(
                x_ii, flow2.permute(0, 2, 3, 1), "bilinear"
            )  # frame i-2 aligned towards i
            x_iii_warped = flow_warp(
                x_iii, flow3.permute(0, 2, 3, 1), "bilinear"
            )  # frame i-3 aligned towards i
            x_forward.append(
                self.pa_deform(
                    torch.cat([x_i, x_ii, x_iii], 1),
                    [x_i_warped, x_ii_warped, x_iii_warped],
                    x[:, i + 1, ...],
                    [flow1, flow2, flow3],
                )
            )

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]


@store_hyperparameters()
class VRT(nn.Module):
    hyperparameters = {}

    """Video Restoration Transformer (VRT).
        A PyTorch impl of : `VRT: A Video Restoration Transformer`  -
          https://arxiv.org/pdf/2201.00000

    Args:
        upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        out_chans (int): Number of output image channels. Default: 3.
        img_size (int | tuple(int)): Size of input image. Default: [6, 64, 64].
        window_size (int | tuple(int)): Window size. Default: (6,8,8).
        depths (list[int]): Depths of each Transformer stage.
        indep_reconsts (list[int]): Layers that extract features of different frames independently.
        embed_dims (list[int]): Number of linear projection output channels.
        num_heads (list[int]): Number of attention head of each stage.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
        spynet_path (str): Pretrained SpyNet model path.
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        recal_all_flows (bool): If True, derive (t,t+2) and (t,t+3) flows from (t,t+1). Default: False.
        nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
        no_checkpoint_attn_blocks (list[int]): Layers without torch.checkpoint for attention modules.
        no_checkpoint_ffn_blocks (list[int]): Layers without torch.checkpoint for feed-forward modules.
    """

    def __init__(
        self,
        upscale=1,
        in_chans=3,
        out_chans=3,
        img_size=[6, 64, 64],
        window_size: Sequence[int] = [6, 8, 8],
        depths: Sequence[int] = [8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
        indep_reconsts: Sequence[int] = [11, 12],
        embed_dims: Sequence[int] = [
            120,
            120,
            120,
            120,
            120,
            120,
            120,
            180,
            180,
            180,
            180,
            180,
            180,
        ],
        num_heads: Sequence[int] = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        mul_attn_ratio=0.75,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        pa_frames=2,
        deformable_groups=16,
        nonblind_denoising=False,
        use_checkpoint_attn=False,
        use_checkpoint_ffn=False,
        no_checkpoint_attn_blocks=[],
        no_checkpoint_ffn_blocks=[],
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.nonblind_denoising = nonblind_denoising

        # conv_first
        if self.pa_frames:
            if self.nonblind_denoising:
                conv_first_in_chans = in_chans * (1 + 2 * 4) + 1
            else:
                conv_first_in_chans = in_chans * (1 + 2 * 4)
        else:
            conv_first_in_chans = in_chans
        self.conv_first = nn.Conv3d(
            conv_first_in_chans, embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

        # main body
        if self.pa_frames:
            self.spynet = SpyNet([2, 3, 4, 5])
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        reshapes = ["none", "down", "down", "down", "up", "up", "up"]
        scales = [1, 2, 4, 8, 4, 2, 1]
        use_checkpoint_attns = [
            False if i in no_checkpoint_attn_blocks else use_checkpoint_attn
            for i in range(len(depths))
        ]
        use_checkpoint_ffns = [
            False if i in no_checkpoint_ffn_blocks else use_checkpoint_ffn
            for i in range(len(depths))
        ]

        # stage 1- 7
        def create_stage(i: int):
            return Stage(
                in_dim=embed_dims[i - 1],
                dim=embed_dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                mul_attn_ratio=mul_attn_ratio,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                pa_frames=pa_frames,
                deformable_groups=deformable_groups,
                reshape=reshapes[i],
                max_residue_magnitude=10 / scales[i],
                use_checkpoint_attn=use_checkpoint_attns[i],
                use_checkpoint_ffn=use_checkpoint_ffns[i],
            )

        self.stage1 = create_stage(0)
        self.stage2 = create_stage(1)
        self.stage3 = create_stage(2)
        self.stage4 = create_stage(3)
        self.stage5 = create_stage(4)
        self.stage6 = create_stage(5)
        self.stage7 = create_stage(6)

        # stage 8
        self.stage8 = nn.ModuleList(
            [
                nn.Sequential(
                    Rearrange("n c d h w ->  n d h w c"),
                    nn.LayerNorm(embed_dims[6]),
                    nn.Linear(embed_dims[6], embed_dims[7]),
                    Rearrange("n d h w c -> n c d h w"),
                )
            ]
        )
        for i in range(7, len(depths)):
            self.stage8.append(
                RTMSA(
                    dim=embed_dims[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=[1, window_size[1], window_size[2]]
                    if i in indep_reconsts
                    else window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    norm_layer=norm_layer,
                    use_checkpoint_attn=use_checkpoint_attns[i],
                    use_checkpoint_ffn=use_checkpoint_ffns[i],
                )
            )

        self.norm = norm_layer(embed_dims[-1])
        self.conv_after_body = nn.Linear(embed_dims[-1], embed_dims[0])

        # reconstruction
        if self.pa_frames:
            if self.upscale == 1:
                # for video deblurring, etc.
                self.conv_last = nn.Conv3d(
                    embed_dims[0], out_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1)
                )
            else:
                # for video sr
                num_feat = 64
                self.conv_before_upsample = nn.Sequential(
                    nn.Conv3d(
                        embed_dims[0],
                        num_feat,
                        kernel_size=(1, 3, 3),
                        padding=(0, 1, 1),
                    ),
                    nn.LeakyReLU(inplace=True),
                )
                self.upsample = Upsample(upscale, num_feat)
                self.conv_last = nn.Conv3d(
                    num_feat, out_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1)
                )
        else:
            num_feat = 64
            self.linear_fuse = nn.Conv2d(
                embed_dims[0] * img_size[0], num_feat, kernel_size=1, stride=1
            )
            self.conv_last = nn.Conv2d(
                num_feat, out_chans, kernel_size=7, stride=1, padding=0
            )

    def reflection_pad2d(self, x, pad=1):
        """Reflection padding for any dtypes (torch.bfloat16.

        Args:
            x: (tensor): BxCxHxW
            pad: (int): Default: 1.
        """

        x = torch.cat(
            [
                torch.flip(x[:, :, 1 : pad + 1, :], [2]),
                x,
                torch.flip(x[:, :, -pad - 1 : -1, :], [2]),
            ],
            2,
        )
        x = torch.cat(
            [
                torch.flip(x[:, :, :, 1 : pad + 1], [3]),
                x,
                torch.flip(x[:, :, :, -pad - 1 : -1], [3]),
            ],
            3,
        )
        return x

    def forward(self, x: torch.Tensor):
        # x: (N, D, C, H, W)

        # main network
        if self.pa_frames:
            # obtain noise level map
            if self.nonblind_denoising:
                x, noise_level_map = (
                    x[:, :, : self.in_chans, :, :],
                    x[:, :, self.in_chans :, :, :],
                )

            x_lq = x.clone()

            # calculate flows
            flows_backward, flows_forward = self.get_flows(x)

            # warp input
            x_backward, x_forward = self.get_aligned_image_2frames(
                x, flows_backward[0], flows_forward[0]
            )
            x = torch.cat([x, x_backward, x_forward], 2)

            # concatenate noise level map
            if self.nonblind_denoising:
                x = torch.cat([x, noise_level_map], 2)  # type: ignore

            if self.upscale == 1:
                # video deblurring, etc.
                x = self.conv_first(x.transpose(1, 2))
                x = x + self.conv_after_body(
                    self.forward_features(x, flows_backward, flows_forward).transpose(
                        1, 4
                    )
                ).transpose(1, 4)
                x = self.conv_last(x).transpose(1, 2)
                return x + x_lq
            else:
                # video sr
                x = self.conv_first(x.transpose(1, 2))
                x = x + self.conv_after_body(
                    self.forward_features(x, flows_backward, flows_forward).transpose(
                        1, 4
                    )
                ).transpose(1, 4)
                x = self.conv_last(
                    self.upsample(self.conv_before_upsample(x))
                ).transpose(1, 2)
                _, _, C, H, W = x.shape
                return x + F.interpolate(
                    x_lq, size=(C, H, W), mode="trilinear", align_corners=False
                )
        else:
            # video fi
            x_mean = x.mean([1, 3, 4], keepdim=True)
            x = x - x_mean

            x = self.conv_first(x.transpose(1, 2))
            x = x + self.conv_after_body(
                self.forward_features(x, [], []).transpose(1, 4)
            ).transpose(1, 4)

            x = torch.cat(torch.unbind(x, 2), 1)
            x = self.conv_last(
                self.reflection_pad2d(F.leaky_relu(self.linear_fuse(x), 0.2), pad=3)
            )
            x = torch.stack(torch.split(x, dim=1, split_size_or_sections=3), 1)

            return x + x_mean

    def get_flows(self, x: torch.Tensor):
        """Get flows for 2 frames, 4 frames or 6 frames."""

        if self.pa_frames == 2:
            flows_backward, flows_forward = self.get_flow_2frames(x)
        elif self.pa_frames == 4:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(
                flows_forward_2frames, flows_backward_2frames
            )
            flows_backward = flows_backward_2frames + flows_backward_4frames
            flows_forward = flows_forward_2frames + flows_forward_4frames
        elif self.pa_frames == 6:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(
                flows_forward_2frames, flows_backward_2frames
            )
            flows_backward_6frames, flows_forward_6frames = self.get_flow_6frames(
                flows_forward_2frames,
                flows_backward_2frames,
                flows_forward_4frames,
                flows_backward_4frames,
            )
            flows_backward = (
                flows_backward_2frames + flows_backward_4frames + flows_backward_6frames
            )
            flows_forward = (
                flows_forward_2frames + flows_forward_4frames + flows_forward_6frames
            )
        else:
            raise ValueError("pa_frames should be 2, 4 or 6.")

        return flows_backward, flows_forward

    def get_flow_2frames(self, x: torch.Tensor):
        """Get flow between frames t and t+1 from x."""

        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        # backward
        flows_backward = self.spynet(x_1, x_2)
        flows_backward = [
            flow.view(b, n - 1, 2, h // (2**i), w // (2**i))
            for flow, i in zip(flows_backward, range(4))
        ]

        # forward
        flows_forward = self.spynet(x_2, x_1)
        flows_forward = [
            flow.view(b, n - 1, 2, h // (2**i), w // (2**i))
            for flow, i in zip(flows_forward, range(4))
        ]

        return flows_backward, flows_forward

    def get_flow_4frames(self, flows_forward, flows_backward):
        """Get flow between t and t+2 from (t,t+1) and (t+1,t+2)."""

        # backward
        d = flows_forward[0].shape[1]
        flows_backward2 = []
        for flows in flows_backward:
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows[:, i - 1, :, :, :]  # flow from i+1 to i
                flow_n2 = flows[:, i, :, :, :]  # flow from i+2 to i+1
                flow_list.insert(
                    0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                )  # flow from i+2 to i
            flows_backward2.append(torch.stack(flow_list, 1))

        # forward
        flows_forward2 = []
        for flows in flows_forward:
            flow_list = []
            for i in range(1, d):
                flow_n1 = flows[:, i, :, :, :]  # flow from i-1 to i
                flow_n2 = flows[:, i - 1, :, :, :]  # flow from i-2 to i-1
                flow_list.append(
                    flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                )  # flow from i-2 to i
            flows_forward2.append(torch.stack(flow_list, 1))

        return flows_backward2, flows_forward2

    def get_flow_6frames(
        self, flows_forward, flows_backward, flows_forward2, flows_backward2
    ):
        """Get flow between t and t+3 from (t,t+2) and (t+2,t+3)."""

        # backward
        d = flows_forward2[0].shape[1]
        flows_backward3 = []
        for flows, flows2 in zip(flows_backward, flows_backward2):
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i+2 to i
                flow_n2 = flows[:, i + 1, :, :, :]  # flow from i+3 to i+2
                flow_list.insert(
                    0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                )  # flow from i+3 to i
            flows_backward3.append(torch.stack(flow_list, 1))

        # forward
        flows_forward3 = []
        for flows, flows2 in zip(flows_forward, flows_forward2):
            flow_list = []
            for i in range(2, d + 1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i-2 to i
                flow_n2 = flows[:, i - 2, :, :, :]  # flow from i-3 to i-2
                flow_list.append(
                    flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                )  # flow from i-3 to i
            flows_forward3.append(torch.stack(flow_list, 1))

        return flows_backward3, flows_forward3

    def get_aligned_image_2frames(
        self, x: torch.Tensor, flows_backward: torch.Tensor, flows_forward: torch.Tensor
    ):
        """Parallel feature warping for 2 frames."""

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[:, i - 1, ...]
            x_backward.insert(
                0, flow_warp(x_i, flow.permute(0, 2, 3, 1), "nearest4")
            )  # frame i+1 aligned towards i

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[:, i, ...]
            x_forward.append(
                flow_warp(x_i, flow.permute(0, 2, 3, 1), "nearest4")
            )  # frame i-1 aligned towards i

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def forward_features(self, x: torch.Tensor, flows_backward, flows_forward):
        """Main network for feature extraction."""

        x1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4])
        x2 = self.stage2(x1, flows_backward[1::4], flows_forward[1::4])
        x3 = self.stage3(x2, flows_backward[2::4], flows_forward[2::4])
        x4 = self.stage4(x3, flows_backward[3::4], flows_forward[3::4])
        x = self.stage5(x4, flows_backward[2::4], flows_forward[2::4])
        x = self.stage6(x + x3, flows_backward[1::4], flows_forward[1::4])
        x = self.stage7(x + x2, flows_backward[0::4], flows_forward[0::4])
        x = x + x1

        for layer in self.stage8:
            x = layer(x)

        x = rearrange(x, "n c d h w -> n d h w c")
        x = self.norm(x)
        x = rearrange(x, "n d h w c -> n c d h w")

        return x
