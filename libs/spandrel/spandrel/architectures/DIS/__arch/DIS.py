from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from spandrel.util import store_hyperparameters


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv - much faster than regular conv"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class FastResBlock(nn.Module):
    """Ultra-fast residual block with minimal operations"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        # Using PReLU - single param per channel, FP16 safe
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class LightBlock(nn.Module):
    """Lightweight block using depthwise separable convolutions"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw_conv = DepthwiseSeparableConv(channels, channels, 3)
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.dw_conv(x))


class PixelShuffleUpsampler(nn.Module):
    """Efficient upsampling using pixel shuffle (ESPCN style)"""

    def __init__(self, in_channels: int, out_channels: int, scale: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale**2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pixel_shuffle(self.conv(x)))


@store_hyperparameters()
class DIS(nn.Module):
    """
    DIS: Direct Image Supersampling

    Design principles:
    - Minimal depth (fewer layers = faster inference)
    - No batch normalization (better for inference, FP16 stable)
    - PReLU activation (FP16 safe, learnable)
    - Pixel shuffle upsampling (efficient and high quality)
    - Global residual learning (image + learned residual)
    - Mix of regular conv and depthwise separable for speed
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 32,
        num_blocks: int = 4,
        scale: int = 4,
        use_depthwise: bool = False,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Shallow feature extraction
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.head_act = nn.PReLU(num_features)

        # Feature extraction body
        if use_depthwise:
            self.body = nn.Sequential(
                *[LightBlock(num_features) for _ in range(num_blocks)]
            )
        else:
            self.body = nn.Sequential(
                *[FastResBlock(num_features) for _ in range(num_blocks)]
            )

        # Feature fusion
        self.fusion = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Upsampling - handle different scales
        if scale == 4:
            self.upsampler = nn.Sequential(
                PixelShuffleUpsampler(num_features, num_features, 2),
                PixelShuffleUpsampler(num_features, num_features, 2),
            )
        elif scale == 3:
            self.upsampler = PixelShuffleUpsampler(num_features, num_features, 3)
        elif scale == 2:
            self.upsampler = PixelShuffleUpsampler(num_features, num_features, 2)
        elif scale == 1:
            self.upsampler = nn.Identity()
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")

        # Final reconstruction
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bilinear upscale for global residual (fast, TensorRT friendly)
        # Using align_corners=False for ONNX compatibility
        if self.scale == 1:
            base = x
        else:
            base = functional.interpolate(
                x, scale_factor=self.scale, mode="bilinear", align_corners=False
            )

        # Feature extraction
        feat = self.head_act(self.head(x))

        # Body with residual
        body_out = self.body(feat)
        body_out = self.fusion(body_out) + feat

        # Upsample and reconstruct
        out = self.upsampler(body_out)
        out = self.tail(out)

        # Global residual learning
        return out + base
