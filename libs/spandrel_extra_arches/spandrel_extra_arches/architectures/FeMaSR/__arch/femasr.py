from __future__ import annotations

import numpy as np
import torch
from torch import nn as nn

from spandrel.architectures.SwinIR.__arch.SwinIR import RSTB
from spandrel.util import store_hyperparameters

from .fema_utils import CombineQuantBlock, ResBlock


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py

    Args:
        n_e : number of embeddings
        e_dim : dimension of embedding
        beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
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


class SwinLayers(nn.Module):
    def __init__(
        self,
        input_resolution=(32, 32),
        embed_dim=256,
        blk_depth=6,
        num_heads=8,
        window_size=8,
    ):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for _i in range(4):
            layer = RSTB(
                embed_dim,
                input_resolution,
                blk_depth,
                num_heads,
                window_size,
                patch_size=1,
            )
            self.swin_blks.append(layer)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        in_channel: int,
        max_depth: int,
        input_res: int = 256,
        channel_query_dict: dict[int, int] = None,  # type: ignore
        norm_type="gn",
        act_type="leakyrelu",
        LQ_stage=True,
    ):
        super().__init__()

        ksz = 3

        self.in_conv = nn.Conv2d(
            in_channel, channel_query_dict[input_res], 4, padding=1
        )

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for _i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        if LQ_stage:
            self.blocks.append(SwinLayers())
            upsampler = nn.ModuleList()
            for _i in range(2):
                in_channel, out_channel = (
                    channel_query_dict[res],
                    channel_query_dict[res * 2],
                )
                upsampler.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                        ResBlock(out_channel, out_channel, norm_type, act_type),
                        ResBlock(out_channel, out_channel, norm_type, act_type),
                    )
                )
                res = res * 2

            self.blocks += upsampler

        self.LQ_stage = LQ_stage

    def forward(self, input: torch.Tensor):
        outputs = []
        x = self.in_conv(input)

        for m in self.blocks:
            x = m(x)
            outputs.append(x)

        return outputs


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_type="gn", act_type="leakyrelu"):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        )

    def forward(self, input):
        return self.block(input)


@store_hyperparameters()
class FeMaSRNet(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        in_channel=3,
        codebook_params=[[32, 1024, 512]],
        gt_resolution=256,
        LQ_stage=False,
        norm_type="gn",
        act_type="silu",
        use_quantize=True,
        scale_factor=1,
        use_residual=True,
    ):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]
        codebook_emb_num = codebook_params[:, 1].astype(int)
        codebook_emb_dim = codebook_params[:, 2].astype(int)

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.scale_factor = scale_factor if LQ_stage else 1
        self.use_residual = use_residual

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

        # build multi-scale vector quantizers
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        for scale in range(0, codebook_params.shape[0]):
            quantize = VectorQuantizer(
                codebook_emb_num[scale],
                codebook_emb_dim[scale],
            )
            self.quantize_group.append(quantize)

            scale_in_ch = channel_query_dict[self.codebook_scale[scale]]
            if scale == 0:
                quant_conv_in_ch = scale_in_ch
                comb_quant_in_ch1 = codebook_emb_dim[scale]
                comb_quant_in_ch2 = 0
            else:
                quant_conv_in_ch = scale_in_ch * 2
                comb_quant_in_ch1 = codebook_emb_dim[scale - 1]
                comb_quant_in_ch2 = codebook_emb_dim[scale]

            self.before_quant_group.append(
                nn.Conv2d(quant_conv_in_ch, codebook_emb_dim[scale], 1)
            )
            self.after_quant_group.append(
                CombineQuantBlock(comb_quant_in_ch1, comb_quant_in_ch2, scale_in_ch)
            )

    def encode_and_decode(self, input):
        enc_feats = self.multiscale_encoder(input.detach())
        if self.LQ_stage:
            enc_feats = enc_feats[-3:]
        else:
            enc_feats = enc_feats[::-1]

        quant_idx = 0
        prev_dec_feat = None
        prev_quant_feat = None
        x = enc_feats[0]
        for i in range(self.max_depth):
            cur_res = self.gt_res // 2**self.max_depth * 2**i
            if cur_res in self.codebook_scale:  # needs to perform quantize
                if prev_dec_feat is not None:
                    before_quant_feat = torch.cat((enc_feats[i], prev_dec_feat), dim=1)
                else:
                    before_quant_feat = enc_feats[i]
                feat_to_quant = self.before_quant_group[quant_idx](before_quant_feat)

                z_quant = self.quantize_group[quant_idx](feat_to_quant)

                if not self.use_quantize:
                    z_quant = feat_to_quant

                after_quant_feat = self.after_quant_group[quant_idx](
                    z_quant, prev_quant_feat
                )

                quant_idx += 1
                prev_quant_feat = z_quant
                x = after_quant_feat
            else:
                if self.LQ_stage and self.use_residual:
                    x = x + enc_feats[i]
                else:
                    x = x

            x = self.decoder_group[i](x)
            prev_dec_feat = x

        out_img = self.out_conv(x)

        return out_img

    def forward(self, input):
        # in HQ stage, or LQ test stage, no GT indices needed.
        return self.encode_and_decode(input)
