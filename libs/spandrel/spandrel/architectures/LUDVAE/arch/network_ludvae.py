import torch
import torch.nn as nn

from spandrel.util import store_hyperparameters


def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return (
        -0.5
        + logsigma2
        - logsigma1
        + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)
    )


def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0.0, 1.0)
    return torch.exp(logsigma) * eps + mu


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filters_num=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class GauBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.m = nn.Conv2d(in_channels, out_channels, 1)
        self.v = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.m(x), self.v(x)


class ProjBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Conv2d(in_channels + in_channels, out_channels, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.proj(x)


@store_hyperparameters()
class LUDVAE(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        in_channel=3,
        filters_num=128,
    ):
        super().__init__()

        self.inconv = nn.Conv2d(in_channel, filters_num, 1)
        self.inconv_n = nn.Conv2d(in_channel, filters_num, 1)

        self.enc_1 = ConvBlock(filters_num, filters_num, filters_num)
        self.enc_2 = ConvBlock(filters_num, filters_num, filters_num)
        self.enc_3 = ConvBlock(filters_num, filters_num, filters_num)

        self.enc_n_1 = ConvBlock(filters_num, filters_num, filters_num)
        self.enc_n_2 = ConvBlock(filters_num, filters_num, filters_num)
        self.enc_n_3 = ConvBlock(filters_num, filters_num, filters_num)

        self.Gauconv_q_3 = GauBlock(filters_num, filters_num)
        self.Gauconv_q_2 = GauBlock(filters_num, filters_num)
        self.Gauconv_q_1 = GauBlock(filters_num, filters_num)

        self.Gauconv_p_2 = GauBlock(filters_num, filters_num)
        self.Gauconv_p_1 = GauBlock(filters_num, filters_num)

        self.dec_3 = ConvBlock(filters_num, filters_num, filters_num)
        self.dec_2 = ConvBlock(filters_num, filters_num, filters_num)
        self.dec_1 = ConvBlock(filters_num, filters_num, filters_num)

        self.proj_3 = ProjBlock(filters_num, filters_num)
        self.proj_2 = ProjBlock(filters_num, filters_num)
        self.proj_c_2 = ProjBlock(filters_num, filters_num)
        self.proj_n_2 = ProjBlock(filters_num, filters_num)
        self.proj_1 = ProjBlock(filters_num, filters_num)
        self.proj_c_1 = ProjBlock(filters_num, filters_num)
        self.proj_n_1 = ProjBlock(filters_num, filters_num)

        self.outconv = nn.Conv2d(filters_num, in_channel, 1)

    def forward(self, x, hx, label):
        _b, c, h, w = x.shape

        act, act_n = self.encode(x, hx)
        dec, kl_loss = self.decode(act, act_n, label)
        rec_loss = self.distortion_loss(dec, x)

        if label.sum() == 0:
            kl_loss = 0 * kl_loss
        else:
            kl_loss = kl_loss / (label.sum() * c * h * w)

        kl_loss = kl_loss.unsqueeze(0)
        rec_loss = rec_loss.unsqueeze(0)

        return rec_loss, kl_loss

    def encode(self, x, hx):
        hx = self.inconv(hx)
        x = self.inconv_n(x)

        act_1 = self.enc_1(hx)
        act_2 = self.enc_2(act_1)
        act_3 = self.enc_3(act_2)

        act_n_1 = self.enc_n_1(x)
        act_n_2 = self.enc_n_2(act_n_1)
        act_n_3 = self.enc_n_3(act_n_2)

        act = [act_1, act_2, act_3]
        act_n = [act_n_1, act_n_2, act_n_3]

        return act, act_n

    def decode(self, act, act_n, label):
        act_1, act_2, act_3 = act
        act_n_1, act_n_2, act_n_3 = act_n

        qm_3, qv_3 = self.Gauconv_q_3(act_n_3)
        pm_3, pv_3 = torch.zeros_like(qm_3), torch.zeros_like(qv_3)
        enc_n_3 = draw_gaussian_diag_samples(qm_3, qv_3) * label
        kl_3 = gaussian_analytical_kl(qm_3, pm_3, qv_3, pv_3) * label
        dec_3 = self.proj_3(act_3, enc_n_3)

        dec_2 = self.dec_3(dec_3)
        dec_2 = self.proj_c_2(dec_2, act_2)
        qm_2, qv_2 = self.Gauconv_q_2(self.proj_n_2(dec_2, act_n_2))
        pm_2, pv_2 = self.Gauconv_p_2(dec_2)
        enc_n_2 = draw_gaussian_diag_samples(qm_2, qv_2) * label
        kl_2 = gaussian_analytical_kl(qm_2, pm_2, qv_2, pv_2) * label
        dec_2 = self.proj_2(dec_2, enc_n_2)

        dec_1 = self.dec_2(dec_2)
        dec_1 = self.proj_c_1(dec_1, act_1)
        qm_1, qv_1 = self.Gauconv_q_1(self.proj_n_1(dec_1, act_n_1))
        pm_1, pv_1 = self.Gauconv_p_1(dec_1)
        enc_n_1 = draw_gaussian_diag_samples(qm_1, qv_1) * label
        kl_1 = gaussian_analytical_kl(qm_1, pm_1, qv_1, pv_1) * label
        dec_1 = self.proj_1(dec_1, enc_n_1)

        dec_0 = self.dec_1(dec_1)
        dec = self.outconv(dec_0)

        kl_loss = kl_1.sum() + kl_2.sum() + kl_3.sum()

        return dec, kl_loss

    def decode_uncond(self, act, label):
        act_1, act_2, act_3 = act

        pm_3, pv_3 = torch.zeros_like(act_3), torch.zeros_like(act_3)
        enc_n_3 = draw_gaussian_diag_samples(pm_3, pv_3) * label
        dec_3 = self.proj_3(act_3, enc_n_3)

        dec_2 = self.dec_3(dec_3)
        dec_2 = self.proj_c_2(dec_2, act_2)
        pm_2, pv_2 = self.Gauconv_p_2(dec_2)
        enc_n_2 = draw_gaussian_diag_samples(pm_2, pv_2) * label
        dec_2 = self.proj_2(dec_2, enc_n_2)

        dec_1 = self.dec_2(dec_2)
        dec_1 = self.proj_c_1(dec_1, act_1)
        pm_1, pv_1 = self.Gauconv_p_1(dec_1)
        enc_n_1 = draw_gaussian_diag_samples(pm_1, pv_1) * label
        dec_1 = self.proj_1(dec_1, enc_n_1)

        dec_0 = self.dec_1(dec_1)
        dec = self.outconv(dec_0)

        return dec

    def distortion_loss(self, x, y):
        return nn.MSELoss()(x, y)

    def translate(self, x, hx, label, temperature=1.0):
        act, _act_n = self.encode(x, hx)
        new_label = (1 - label) * temperature
        dec = self.decode_uncond(act, new_label)

        return dec

    def reconstruction(self, x, hx, label):
        act, act_n = self.encode(x, hx)
        dec, kl_loss = self.decode(act, act_n, label)
        rec_loss = self.distortion_loss(dec, x)

        kl_loss = kl_loss.unsqueeze(0)
        rec_loss = rec_loss.unsqueeze(0)

        return dec
