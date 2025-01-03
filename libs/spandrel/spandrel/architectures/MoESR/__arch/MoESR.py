import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.nn.init import trunc_normal_

from spandrel.architectures.__arch_helpers.dysample import DySample
from spandrel.util import store_hyperparameters

SampleMods = Literal[
    "conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv", "dysample"
]


class UniUpsample(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods,
        scale: int = 2,
        in_dim: int = 64,
        out_dim: int = 3,
        mid_dim: int = 64,  # Only pixelshuffle
        group: int = 4,  # Only DySample
    ) -> None:
        m = []

        if scale == 1 or upsample == "conv":
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "pixelshuffledirect":
            m.extend(
                [nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)]
            )
        elif upsample == "pixelshuffle":
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)]
                    )
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "dysample":
            m.append(DySample(in_dim, out_dim, scale, group))
        else:
            raise ValueError(
                f"An invalid Upsample was selected. Please choose one of {SampleMods}"
            )
        super().__init__(*m)

        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [
                    1,  # Block version, if you change something, please number from the end so that you can distinguish between authorized changes and third parties
                    list(SampleMods.__args__).index(upsample),  # pyright: ignore  # UpSample method index
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                ],
                dtype=torch.uint8,
            ),
        )


class LayerNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class InceptionDWConv2d(nn.Module):
    """Inception depthweise convolution"""

    def __init__(
        self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125
    ) -> None:
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.split_indexes = [in_channels - 3 * gc, gc, gc, gc]

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 8 / 3,
        conv_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)

        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]

        self.conv = InceptionDWConv2d(conv_channels)
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        return (x * self.gamma) + shortcut


class MSG(nn.Module):
    def __init__(self, dim, expansion_msg=1.5) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.PixelUnshuffle(2),
            nn.LeakyReLU(0.1, True),
        )
        self.gated = nn.Sequential(
            *[GatedCNNBlock(dim, expansion_ratio=expansion_msg) for _ in range(3)]
        )
        self.up = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        out = self.down(x)
        out = self.gated(out)
        return self.up(out) + x


class Blocks(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        blocks: int = 4,
        expansion_factor: float = 1.5,
        expansion_msg: float = 1.5,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[GatedCNNBlock(dim, expansion_factor) for _ in range(blocks)]
        )
        self.msg = MSG(dim, expansion_msg)

    def forward(self, x):
        return self.msg(self.blocks(x))


@store_hyperparameters()
class MoESR(nn.Module):
    """Mamba out Excitation Super-Resolution"""

    hyperparameters = {}

    def __init__(
        self,
        *,
        in_ch: int = 3,
        out_ch: int = 3,
        scale: int = 4,
        dim: int = 64,
        n_blocks: int = 9,
        n_block: int = 4,
        expansion_factor: float = 2.65625,
        expansion_msg: float = 1.5,
        upsampler: SampleMods = "pixelshuffledirect",
        upsample_dim: int = 64,
    ) -> None:
        super().__init__()
        if upsampler == "conv":
            scale = 1
        self.scale = scale
        self.in_to_dim = nn.Conv2d(in_ch, dim, 3, 1, 1)
        self.blocks = nn.Sequential(
            *[
                Blocks(dim, n_block, expansion_factor, expansion_msg)
                for _ in range(n_blocks)
            ]
        )
        self.upscale = UniUpsample(upsampler, scale, dim, out_ch, upsample_dim)

    def check_img_size(self, x, resolution):
        h, w = resolution
        scaled_size = 2
        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x):
        _b, _c, h, w = x.shape
        x = self.check_img_size(x, (h, w))
        x = self.in_to_dim(x)
        x = self.blocks(x) + x
        return self.upscale(x)[:, :, : h * self.scale, : w * self.scale]
