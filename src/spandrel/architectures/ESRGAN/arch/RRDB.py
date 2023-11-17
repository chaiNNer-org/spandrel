#!/usr/bin/env python3
# type: ignore
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...__arch_helpers import block as B


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        num_filters: int = 64,
        num_blocks: int = 23,
        scale: int = 4,
        c2x2: bool = False,
        plus: bool = False,
        shuffle_factor: int | None = None,
        norm=None,
        act: str = "leakyrelu",
        upsampler: str = "upconv",
        mode: B.ConvMode = "CNA",
    ) -> None:
        """
        ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
        By Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao,
        and Chen Change Loy.
        This is old-arch Residual in Residual Dense Block Network and is not
        the newest revision that's available at github.com/xinntao/ESRGAN.
        This is on purpose, the newest Network has severely limited the
        potential use of the Network with no benefits.
        This network supports model files from both new and old-arch.
        Args:
            norm: Normalization layer
            act: Activation layer
            upsampler: Upsample layer. upconv, pixel_shuffle
            mode: Convolution mode
        """
        super().__init__()

        self.shuffle_factor = shuffle_factor
        self.scale = scale

        upsample_block = {
            "upconv": B.upconv_block,
            "pixel_shuffle": B.pixelshuffle_block,
        }.get(upsampler)
        if upsample_block is None:
            raise NotImplementedError(f"Upsample mode [{upsampler}] is not found")

        if scale == 3:
            upsample_blocks = upsample_block(
                in_nc=num_filters,
                out_nc=num_filters,
                upscale_factor=3,
                act_type=act,
                c2x2=c2x2,
            )
        else:
            upsample_blocks = [
                upsample_block(
                    in_nc=num_filters,
                    out_nc=num_filters,
                    act_type=act,
                    c2x2=c2x2,
                )
                for _ in range(int(math.log(scale, 2)))
            ]

        self.model = B.sequential(
            # fea conv
            B.conv_block(
                in_nc=in_nc,
                out_nc=num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
            B.ShortcutBlock(
                B.sequential(
                    # rrdb blocks
                    *[
                        B.RRDB(
                            nf=num_filters,
                            kernel_size=3,
                            gc=32,
                            stride=1,
                            bias=True,
                            pad_type="zero",
                            norm_type=norm,
                            act_type=act,
                            mode="CNA",
                            plus=plus,
                            c2x2=c2x2,
                        )
                        for _ in range(num_blocks)
                    ],
                    # lr conv
                    B.conv_block(
                        in_nc=num_filters,
                        out_nc=num_filters,
                        kernel_size=3,
                        norm_type=norm,
                        act_type=None,
                        mode=mode,
                        c2x2=c2x2,
                    ),
                )
            ),
            *upsample_blocks,
            # hr_conv0
            B.conv_block(
                in_nc=num_filters,
                out_nc=num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=act,
                c2x2=c2x2,
            ),
            # hr_conv1
            B.conv_block(
                in_nc=num_filters,
                out_nc=out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
        )

    def forward(self, x):
        if self.shuffle_factor:
            _, _, h, w = x.size()
            mod_pad_h = (
                self.shuffle_factor - h % self.shuffle_factor
            ) % self.shuffle_factor
            mod_pad_w = (
                self.shuffle_factor - w % self.shuffle_factor
            ) % self.shuffle_factor
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
            x = torch.pixel_unshuffle(x, downscale_factor=self.shuffle_factor)
            x = self.model(x)
            return x[:, :, : h * self.scale, : w * self.scale]
        return self.model(x)
