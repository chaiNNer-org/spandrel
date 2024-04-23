import torch.nn as nn

from spandrel.util import store_hyperparameters

from .common import (
    ConvNorm,
    InOutPaddings,
    Interpolation,
    ResBlock,
    UpConvNorm,
    conv7x7,
    sub_mean,
)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, nf_start=32, norm=False):
        super().__init__()

        nf = nf_start
        relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.body = nn.Sequential(
            ConvNorm(in_channels, nf * 1, 7, stride=1, norm=norm),
            relu,
            ConvNorm(nf * 1, nf * 2, 5, stride=2, norm=norm),
            relu,
            ConvNorm(nf * 2, nf * 4, 5, stride=2, norm=norm),
            relu,
            ConvNorm(nf * 4, nf * 6, 5, stride=2, norm=norm),
        )

        self.interpolate = Interpolation(5, 12, nf * 6, reduction=16, act=relu)

    def forward(self, x1, x2):
        """
        Encoder: Feature Extraction --> Feature Fusion --> Return
        """
        feats1 = self.body(x1)
        feats2 = self.body(x2)

        feats = self.interpolate(feats1, feats2)

        return feats


class Decoder(nn.Module):
    def __init__(self, in_channels=192, out_channels=3, norm=False, up_mode="shuffle"):
        super().__init__()

        relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        nf = [in_channels, (in_channels * 2) // 3, in_channels // 3, in_channels // 6]
        # nf = [192, 128, 64, 32]
        # nf = [186, 124, 62, 31]
        self.body = nn.Sequential(
            UpConvNorm(nf[0], nf[1], mode=up_mode, norm=norm),
            ResBlock(nf[1], nf[1], act=relu),
            UpConvNorm(nf[1], nf[2], mode=up_mode, norm=norm),
            ResBlock(nf[2], nf[2], act=relu),
            UpConvNorm(nf[2], nf[3], mode=up_mode, norm=norm),
            ResBlock(nf[3], nf[3], act=relu),
            conv7x7(nf[3], out_channels),
        )

    def forward(self, feats):
        out = self.body(feats)
        # out = self.conv_final(out)

        return out


@store_hyperparameters(extra_parameters={"kind": "CAIN_EncDec"})
class CAIN_EncDec(nn.Module):
    hyperparameters = {}

    def __init__(self, start_filters=32, up_mode="shuffle"):
        super().__init__()

        self.encoder = Encoder(in_channels=3, norm=False)
        self.decoder = Decoder(
            in_channels=start_filters * 6, norm=False, up_mode=up_mode
        )

    def forward(self, x1, x2):
        x1, m1 = sub_mean(x1)
        x2, m2 = sub_mean(x2)

        paddingInput, paddingOutput = InOutPaddings(x1)
        x1 = paddingInput(x1)
        x2 = paddingInput(x2)

        feats = self.encoder(x1, x2)
        out = self.decoder(feats)

        out = paddingOutput(out)

        mi = (m1 + m2) / 2
        out += mi

        return out, feats
