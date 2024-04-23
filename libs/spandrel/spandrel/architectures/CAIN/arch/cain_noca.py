import torch
import torch.nn as nn

from spandrel.util import store_hyperparameters

from .common import InOutPaddings, Interpolation_res, PixelShuffle, sub_mean


class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3):
        super().__init__()

        self.shuffler = PixelShuffle(1 / 2**depth)
        self.interpolate = Interpolation_res(5, 12, in_channels * (4**depth))

    def forward(self, x1, x2):
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)

        feats = self.interpolate(feats1, feats2)

        return feats


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super().__init__()

        self.shuffler = PixelShuffle(2**depth)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


@store_hyperparameters(extra_parameters={"kind": "CAIN_NoCA"})
class CAIN_NoCA(nn.Module):
    hyperparameters = {}

    def __init__(self, depth=3):
        super().__init__()
        self.depth = depth

        self.encoder = Encoder(in_channels=3, depth=depth)
        self.decoder = Decoder(depth=depth)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
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
