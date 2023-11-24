import torch
from torch import nn as nn
from torch.nn import functional as F


class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """

    def __init__(self, channels, norm_type="bn"):
        super().__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        self.channels = channels
        if norm_type == "bn":
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == "in":
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == "gn":
            self.norm = nn.GroupNorm(
                num_groups=32, num_channels=channels, eps=1e-6, affine=True
            )
        elif norm_type == "none":
            self.norm = lambda x: x * 1.0
        else:
            assert False, f"Norm type {norm_type} not support."

    def forward(self, x):
        return self.norm(x)


class ActLayer(nn.Module):
    """activation layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu
            - SELU
            - none: direct pass
    """

    def __init__(self, channels, relu_type="leakyrelu"):
        super().__init__()
        relu_type = relu_type.lower()
        if relu_type == "relu":
            self.func = nn.ReLU(True)
        elif relu_type == "leakyrelu":
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == "prelu":
            self.func = nn.PReLU(channels)
        elif relu_type == "none":
            self.func = lambda x: x * 1.0
        elif relu_type == "silu":
            self.func = nn.SiLU(True)
        elif relu_type == "gelu":
            self.func = nn.GELU()
        else:
            assert False, f"activation type {relu_type} not support."

    def forward(self, x):
        return self.func(x)


class ResBlock(nn.Module):
    """
    Use preactivation version of residual block, the same as taming
    """

    def __init__(self, in_channel, out_channel, norm_type="gn", act_type="leakyrelu"):
        super().__init__()

        self.conv = nn.Sequential(
            NormLayer(in_channel, norm_type),
            ActLayer(in_channel, act_type),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
        )

    def forward(self, input):
        res = self.conv(input)
        out = res + input
        return out


class CombineQuantBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_ch1 + in_ch2, out_channel, 3, 1, 1)

    def forward(self, input1, input2=None):
        if input2 is not None:
            input2 = F.interpolate(input2, input1.shape[2:])
            input = torch.cat((input1, input2), dim=1)
        else:
            input = input1
        out = self.conv(input)
        return out
