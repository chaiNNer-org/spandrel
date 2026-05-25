import torch
import torch.nn as nn


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):  # type: ignore
        ctx.eps = eps
        _N, C, _H, _W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        eps = ctx.eps

        _N, C, _H, _W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))  # type: ignore
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))  # type: ignore
        self.eps = eps

    def forward(self, x):
        # Compute statistics in fp32. In fp16 the original path overflowed
        # ((x-mu)**2 exceeds 65504) and eps=1e-6 fell below fp16 resolution,
        # giving sqrt(~0) -> garbage output. fp32 stats keep fp16 numerically
        # sound (and this is ONNX-exportable, unlike the custom autograd op).
        c = x.shape[1]
        xf = x.float()
        mu = xf.mean(1, keepdim=True)
        var = (xf - mu).pow(2).mean(1, keepdim=True)
        y = ((xf - mu) / torch.sqrt(var + self.eps)).to(x.dtype)
        return self.weight.view(1, c, 1, 1) * y + self.bias.view(1, c, 1, 1)
