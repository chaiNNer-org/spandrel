from __future__ import annotations

import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F

from . import thops


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels: int, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.linalg.qr(torch.randn(*w_shape))[0].float()
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(w_init))
        else:
            plu: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = torch.linalg.lu(
                w_init
            )
            np_p, np_l, np_u = plu
            np_s = torch.diag(np_u)
            np_sign_s = torch.sign(np_s)
            np_log_s = torch.log(torch.abs(np_s))
            np_u = torch.triu(np_u, diagonal=1)
            l_mask = torch.tril(torch.ones(w_shape, dtype=torch.float32), -1)
            eye = torch.eye(*w_shape, dtype=torch.float32)

            self.register_buffer("p", np_p.float())
            self.register_buffer("sign_s", (np_sign_s.float()))
            self.l = nn.Parameter(np_l.float())
            self.log_s = nn.Parameter(np_log_s.float())
            self.u = nn.Parameter(np_u.float())
            self.l_mask = l_mask
            self.eye = eye
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = thops.pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = (
                    torch.inverse(self.weight.double())
                    .float()
                    .view(w_shape[0], w_shape[1], 1, 1)
                )
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)  # type: ignore
            self.sign_s = self.sign_s.to(input.device)  # type: ignore
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye  # noqa: E741
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(
                self.sign_s * torch.exp(self.log_s)
            )
            dlogdet = thops.sum(self.log_s) * thops.pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()  # noqa: E741
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet
