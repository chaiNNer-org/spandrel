import torch
import torch.nn as nn

from .common import GCM, SDM, DenseProjection, ResBlock, ResidualGroup, default_conv


class SGNet(nn.Module):
    def __init__(self, num_feats: int, kernel_size: int, scale: int):
        super().__init__()
        self.conv_rgb1 = nn.Conv2d(
            in_channels=3, out_channels=num_feats, kernel_size=kernel_size, padding=1
        )
        self.rgb_rb2 = ResBlock(
            default_conv,
            num_feats,
            kernel_size,
            bias=True,
            bn=False,
            act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            res_scale=1,
        )
        self.rgb_rb3 = ResBlock(
            default_conv,
            num_feats,
            kernel_size,
            bias=True,
            bn=False,
            act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            res_scale=1,
        )
        self.rgb_rb4 = ResBlock(
            default_conv,
            num_feats,
            kernel_size,
            bias=True,
            bn=False,
            act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            res_scale=1,
        )

        self.conv_dp1 = nn.Conv2d(
            in_channels=1, out_channels=num_feats, kernel_size=kernel_size, padding=1
        )
        self.conv_dp2 = nn.Conv2d(
            in_channels=num_feats,
            out_channels=2 * num_feats,
            kernel_size=kernel_size,
            padding=1,
        )
        self.dp_rg1 = ResidualGroup(
            default_conv, num_feats, kernel_size, reduction=16, n_resblocks=6
        )
        self.dp_rg2 = ResidualGroup(
            default_conv, 2 * num_feats, kernel_size, reduction=16, n_resblocks=6
        )
        self.dp_rg3 = ResidualGroup(
            default_conv, 2 * num_feats, kernel_size, reduction=16, n_resblocks=6
        )
        self.dp_rg4 = ResidualGroup(
            default_conv, 3 * num_feats, kernel_size, reduction=16, n_resblocks=6
        )

        self.bridge1 = SDM(channels=num_feats, rgb_channels=num_feats)
        self.bridge2 = SDM(channels=2 * num_feats, rgb_channels=num_feats)
        self.bridge3 = SDM(channels=3 * num_feats, rgb_channels=num_feats)

        self.c_de = default_conv(4 * num_feats, 2 * num_feats, 1)

        my_tail = [
            ResidualGroup(
                default_conv, 3 * num_feats, kernel_size, reduction=16, n_resblocks=8
            ),
            ResidualGroup(
                default_conv, 3 * num_feats, kernel_size, reduction=16, n_resblocks=8
            ),
            ResidualGroup(
                default_conv, 3 * num_feats, kernel_size, reduction=16, n_resblocks=8
            ),
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(
            3 * num_feats, 3 * num_feats, scale, up=True, bottleneck=False
        )
        last_conv = [
            default_conv(3 * num_feats, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True),
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode="bicubic")

        self.c_rd = default_conv(8 * num_feats, 3 * num_feats, 1)
        self.c_grad = default_conv(2 * num_feats, num_feats, 1)
        self.c_grad2 = default_conv(3 * num_feats, 2 * num_feats, 1)
        self.c_grad3 = default_conv(3 * num_feats, 3 * num_feats, 1)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.gradNet = GCM(n_feats=num_feats)

    def forward(self, x):
        image, depth = x

        out_re, grad_d4 = self.gradNet(depth, image)

        dp_in = self.act(self.conv_dp1(depth))
        dp1 = self.dp_rg1(dp_in)

        cat10 = torch.cat([dp1, grad_d4], dim=1)
        dp1_ = self.c_grad(cat10)

        rgb1 = self.act(self.conv_rgb1(image))
        rgb2 = self.rgb_rb2(rgb1)

        ca1_in, r1 = self.bridge1(dp1_, rgb2)
        dp2 = self.dp_rg2(torch.cat([dp1, ca1_in + dp_in], 1))

        cat11 = torch.cat([dp2, grad_d4], dim=1)
        dp2_ = self.c_grad2(cat11)

        rgb3 = self.rgb_rb3(r1)
        ca2_in, r2 = self.bridge2(dp2_, rgb3)

        ca2_in_ = ca2_in + self.conv_dp2(dp_in)

        cat1_0 = torch.cat([dp2, ca2_in_], 1)

        dp3 = self.dp_rg3(self.c_de(cat1_0))
        rgb4 = self.rgb_rb4(r2)

        cat12 = torch.cat([dp3, grad_d4], dim=1)
        dp3_ = self.c_grad3(cat12)

        ca3_in, _r3 = self.bridge3(dp3_, rgb4)

        cat1 = torch.cat([dp1, dp2, dp3, ca3_in], 1)

        dp4 = self.dp_rg4(self.c_rd(cat1))

        tail_in = self.upsampler(dp4)
        out = self.last_conv(self.tail(tail_in))

        out = out + self.bicubic(depth)

        return out, out_re
