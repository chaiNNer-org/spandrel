import torch.nn as nn

from spandrel.util import store_hyperparameters

from .basicblock import conv, sequential

"""
# --------------------------------------------
# DnCNN (20 conv layers)
# FDnCNN (20 conv layers)
# --------------------------------------------
# References:
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}
# --------------------------------------------
"""


# --------------------------------------------
# DnCNN
# --------------------------------------------
@store_hyperparameters()
class DnCNN(nn.Module):
    hyperparameters = {}

    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode="BR"):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super().__init__()
        assert (
            "R" in act_mode or "L" in act_mode
        ), "Examples of activation function: R, L, BR, BL, IR, IL"
        bias = True

        m_head = conv(in_nc, nc, mode="C" + act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode="C" + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = conv(nc, out_nc, mode="C", bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x - n


@store_hyperparameters()
class IRCNN(nn.Module):
    hyperparameters = {}

    def __init__(self, in_nc=1, out_nc=1, nc=64):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super().__init__()
        L = []
        L.append(
            nn.Conv2d(
                in_channels=in_nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=out_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True,
            )
        )
        self.model = sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x - n


# --------------------------------------------
# FDnCNN
# --------------------------------------------
# Compared with DnCNN, FDnCNN has three modifications:
# 1) add noise level map as input
# 2) remove residual learning and BN
# 3) train with L1 loss
# may need more training time, but will not reduce the final PSNR too much.
# --------------------------------------------
@store_hyperparameters()
class FDnCNN(nn.Module):
    hyperparameters = {}

    def __init__(self, in_nc=2, out_nc=1, nc=64, nb=20, act_mode="R"):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        """
        super().__init__()
        assert (
            "R" in act_mode or "L" in act_mode
        ), "Examples of activation function: R, L, BR, BL, IR, IL"
        bias = True

        m_head = conv(in_nc, nc, mode="C" + act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode="C" + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = conv(nc, out_nc, mode="C", bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x
