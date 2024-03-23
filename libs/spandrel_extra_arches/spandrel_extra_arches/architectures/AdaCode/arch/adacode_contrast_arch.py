from __future__ import annotations

import numpy as np
import torch
from torch import nn as nn

from spandrel.util import store_hyperparameters

from ...FeMaSR.arch.femasr import DecoderBlock, MultiScaleEncoder, SwinLayers


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def dist(self, x, y):
        return (
            torch.sum(x**2, dim=1, keepdim=True)
            + torch.sum(y**2, dim=1)
            - 2 * torch.matmul(x, y.t())
        )

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization.
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], codebook.shape[0]
        ).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class WeightPredictor(nn.Module):
    def __init__(
        self,
        in_channel: int,
        cls: int,
        weight_softmax=False,
        **swin_opts,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(SwinLayers(**swin_opts))
        # weight
        self.blocks.append(nn.Conv2d(in_channel, cls, kernel_size=1))
        if weight_softmax:
            self.blocks.append(nn.Softmax(dim=1))

    def forward(self, input: torch.Tensor):
        x = input
        for m in self.blocks:
            x = m(x)
        return x


@store_hyperparameters()
class AdaCodeSRNet_Contrast(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        in_channel=3,
        codebook_params: list[list[int]] = [[32, 256, 256]],
        gt_resolution=256,
        LQ_stage=False,
        norm_type="gn",
        act_type="silu",
        use_quantize=True,
        scale_factor=1,
        use_residual=True,
        weight_softmax=False,
    ):
        super().__init__()

        codebook_params_np = np.array(codebook_params)

        self.codebook_scale = codebook_params_np[:, 0]
        codebook_emb_num = codebook_params_np[:, 1].astype(int)
        codebook_emb_dim = codebook_params_np[:, 2].astype(int)

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.scale_factor = scale_factor if LQ_stage else 1
        self.use_residual = use_residual
        self.weight_softmax = weight_softmax

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        # build encoder
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        encode_depth = int(
            np.log2(gt_resolution // self.scale_factor // self.codebook_scale[0])
        )
        self.multiscale_encoder = MultiScaleEncoder(
            in_channel,
            encode_depth,
            self.gt_res // self.scale_factor,
            channel_query_dict,
            norm_type,
            act_type,
            LQ_stage,
        )

        # build decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2**self.max_depth * 2**i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)  # type: ignore

        # build weight predictor
        self.weight_predictor = WeightPredictor(
            channel_query_dict[self.codebook_scale[0]],
            self.codebook_scale.shape[0],
            self.weight_softmax,
        )

        # build multi-scale vector quantizers
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        for i in range(codebook_params_np.shape[0]):
            quantize = VectorQuantizer(
                codebook_emb_num[i],
                codebook_emb_dim[i],
            )
            self.quantize_group.append(quantize)

            quant_in_ch = channel_query_dict[self.codebook_scale[i]]
            self.before_quant_group.append(
                nn.Conv2d(quant_in_ch, codebook_emb_dim[i], 1)
            )
            self.after_quant_group.append(
                nn.Conv2d(codebook_emb_dim[i], quant_in_ch, 3, 1, 1)
            )

    def encode_and_decode(self, input):
        enc_feats = self.multiscale_encoder(input.detach())
        if self.LQ_stage:
            enc_feats = enc_feats[-3:]
        else:
            enc_feats = enc_feats[::-1]

        after_quant_feat_group = []
        x = enc_feats[0]
        for i in range(self.max_depth):
            cur_res = self.gt_res // 2**self.max_depth * 2**i
            if cur_res in self.codebook_scale:  # needs to perform quantize
                before_quant_feat = enc_feats[i]

                # quantize features with multiple codebooks
                for codebook_idx in range(self.codebook_scale.shape[0]):
                    feat_to_quant = self.before_quant_group[codebook_idx](
                        before_quant_feat
                    )

                    z_quant = self.quantize_group[codebook_idx](feat_to_quant)

                    if not self.use_quantize:
                        z_quant = feat_to_quant

                    after_quant_feat = self.after_quant_group[codebook_idx](z_quant)
                    after_quant_feat_group.append(after_quant_feat)

                # merge feature tensors
                weight = self.weight_predictor(before_quant_feat).unsqueeze(
                    2
                )  # B x N x 1 x H x W
                x = torch.sum(
                    torch.mul(
                        torch.stack(after_quant_feat_group).transpose(0, 1), weight
                    ),
                    dim=1,
                )
            else:
                if self.LQ_stage and self.use_residual:
                    x = x + enc_feats[i]
                else:
                    x = x

            x = self.decoder_group[i](x)

        out_img = self.out_conv(x)

        return out_img

    def forward(self, input):
        # in HQ stage, or LQ test stage, no GT indices needed.
        return self.encode_and_decode(input)
