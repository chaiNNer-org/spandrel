from __future__ import annotations

from typing import Callable

from ..architectures import (
    CRAFT,
    DAT,
    DITN,
    ESRGAN,
    FBCNN,
    GFPGAN,
    GRL,
    HAT,
    MAT,
    SAFMN,
    SPAN,
    SPSR,
    CodeFormer,
    Compact,
    DDColor,
    FeMaSR,
    KBNet,
    LaMa,
    MMRealSR,
    NAFNet,
    OmniSR,
    RealCUGAN,
    RestoreFormer,
    SCUNet,
    SRFormer,
    SwiftSRGAN,
    Swin2SR,
    SwinIR,
    Uformer,
)
from .model_descriptor import StateDict
from .registry import ArchRegistry, ArchSupport


def _has_keys(*keys: str) -> Callable[[StateDict], bool]:
    def _detect(state_dict: StateDict) -> bool:
        return all(key in state_dict for key in keys)

    return _detect


MAIN_REGISTRY = ArchRegistry()
"""
The main architecture registry of spandrel.

Modifying this registry will affect all `ModelLoader` instances without a custom registry.
"""

MAIN_REGISTRY.add(
    ArchSupport(
        # SRVGGNet Real-ESRGAN (v2)
        id="Compact",
        detect=_has_keys("body.0.weight", "body.1.weight"),
        load=Compact.load,
    ),
    ArchSupport(
        # SPSR (ESRGAN with lots of extra layers)
        id="SPSR",
        detect=_has_keys("f_HR_conv1.0.weight", "model.0.weight"),
        load=SPSR.load,
    ),
    ArchSupport(
        id="SwiftSRGAN",
        detect=_has_keys("initial.cnn.depthwise.weight", "final_conv.pointwise.weight"),
        load=SwiftSRGAN.load,
    ),
    ArchSupport(
        id="SRFormer",
        detect=_has_keys(
            "layers.0.residual_group.blocks.0.norm1.weight",
            "layers.0.residual_group.blocks.0.attn.aligned_relative_position_index",
            "conv_first.weight",
            "layers.0.residual_group.blocks.0.mlp.fc1.bias",
            "layers.0.residual_group.blocks.0.attn.aligned_relative_position_index",
        ),
        load=SRFormer.load,
    ),
    ArchSupport(
        id="HAT",
        detect=_has_keys(
            "layers.0.residual_group.blocks.0.norm1.weight",
            "layers.0.residual_group.blocks.0.conv_block.cab.0.weight",
            "conv_last.weight",
            "conv_first.weight",
            "layers.0.residual_group.blocks.0.mlp.fc1.bias",
            "relative_position_index_SA",
        ),
        load=HAT.load,
    ),
    ArchSupport(
        id="GRL",
        detect=lambda state: (
            _has_keys(
                "conv_first.weight",
                "norm_start.weight",
                "norm_end.weight",
                "layers.0.blocks.0.attn.window_attn.attn_transform.logit_scale",
                "layers.0.blocks.0.attn.stripe_attn.attn_transform1.logit_scale",
            )(state)
            or _has_keys(
                "model.conv_first.weight",
                "model.norm_start.weight",
                "model.norm_end.weight",
                "model.layers.0.blocks.0.attn.window_attn.attn_transform.logit_scale",
                "model.layers.0.blocks.0.attn.stripe_attn.attn_transform1.logit_scale",
            )(state)
            or _has_keys(
                "model_g.conv_first.weight",
                "model_g.norm_start.weight",
                "model_g.norm_end.weight",
                "model_g.layers.0.blocks.0.attn.window_attn.attn_transform.logit_scale",
                "model_g.layers.0.blocks.0.attn.stripe_attn.attn_transform1.logit_scale",
            )(state)
        ),
        load=GRL.load,
    ),
    ArchSupport(
        id="Swin2SR",
        detect=_has_keys(
            "layers.0.residual_group.blocks.0.norm1.weight",
            "patch_embed.proj.weight",
            "conv_first.weight",
            "layers.0.residual_group.blocks.0.mlp.fc1.bias",
            "layers.0.residual_group.blocks.0.attn.relative_position_index",
        ),
        load=Swin2SR.load,
    ),
    ArchSupport(
        id="SwinIR",
        detect=_has_keys(
            "layers.0.residual_group.blocks.0.norm1.weight",
            "conv_first.weight",
            "layers.0.residual_group.blocks.0.mlp.fc1.bias",
            "layers.0.residual_group.blocks.0.attn.relative_position_index",
        ),
        load=SwinIR.load,
    ),
    ArchSupport(
        id="FeMaSR",
        detect=_has_keys(
            "multiscale_encoder.in_conv.weight",
            "multiscale_encoder.blocks.0.0.weight",
            "decoder_group.0.block.1.weight",
            "out_conv.weight",
            "before_quant_group.0.weight",
        ),
        load=FeMaSR.load,
    ),
    ArchSupport(
        id="GFPGAN",
        detect=_has_keys("toRGB.0.weight", "stylegan_decoder.style_mlp.1.weight"),
        load=GFPGAN.load,
    ),
    ArchSupport(
        id="RestoreFormer",
        detect=_has_keys(
            "encoder.conv_in.weight", "encoder.down.0.block.0.norm1.weight"
        ),
        load=RestoreFormer.load,
    ),
    ArchSupport(
        id="CodeFormer",
        detect=_has_keys(
            "encoder.blocks.0.weight",
            "quantize.embedding.weight",
            "position_emb",
            "quantize.embedding.weight",
            "ft_layers.0.self_attn.in_proj_weight",
            "encoder.blocks.0.weight",
        ),
        load=CodeFormer.load,
    ),
    ArchSupport(
        id="LaMa",
        detect=lambda state: (
            "model.model.1.bn_l.running_mean" in state
            or "generator.model.1.bn_l.running_mean" in state
        ),
        load=LaMa.load,
    ),
    ArchSupport(
        id="MAT",
        detect=_has_keys("synthesis.first_stage.conv_first.conv.resample_filter"),
        load=MAT.load,
    ),
    ArchSupport(
        id="OmniSR",
        detect=_has_keys(
            "residual_layer.0.residual_layer.0.layer.0.fn.0.weight",
            "input.weight",
            "up.0.weight",
        ),
        load=OmniSR.load,
    ),
    ArchSupport(
        id="SCUNet",
        detect=_has_keys("m_head.0.weight", "m_tail.0.weight"),
        load=SCUNet.load,
    ),
    ArchSupport(
        id="FBCNN",
        detect=_has_keys(
            "m_head.weight",
            "m_down1.0.res.0.weight",
            "m_down2.0.res.0.weight",
            "m_body_encoder.0.res.0.weight",
            "m_tail.weight",
            "qf_pred.0.res.0.weight",
        ),
        load=FBCNN.load,
    ),
    ArchSupport(
        id="Uformer",
        detect=_has_keys(
            "input_proj.proj.0.weight",
            "output_proj.proj.0.weight",
            "encoderlayer_0.blocks.0.norm1.weight",
            "encoderlayer_2.blocks.0.norm1.weight",
            "conv.blocks.0.norm1.weight",
            "decoderlayer_0.blocks.0.norm1.weight",
            "decoderlayer_2.blocks.0.norm1.weight",
        ),
        load=Uformer.load,
    ),
    ArchSupport(
        id="DAT",
        detect=_has_keys("layers.0.blocks.2.attn.attn_mask_0", "conv_first.weight"),
        load=DAT.load,
    ),
    ArchSupport(
        id="CRAFT",
        detect=_has_keys(
            "conv_first.weight", "layers.0.residual_group.hf_blocks.0.attn.temperature"
        ),
        load=CRAFT.load,
    ),
    ArchSupport(
        id="KBNet",
        detect=lambda state: (
            # KBNet_s
            _has_keys(
                "intro.weight",
                "encoders.0.0.attgamma",
                "middle_blks.0.w",
                "decoders.0.0.attgamma",
                "ending.weight",
            )(state)
            # KBNet_l
            or _has_keys(
                "patch_embed.proj.weight",
                "encoder_level3.0.ffn.project_out.weight",
                "latent.0.ffn.qkv.weight",
                "refinement.0.attn.dwconv.0.weight",
            )(state)
        ),
        load=KBNet.load,
    ),
    ArchSupport(
        id="DITN",
        detect=_has_keys(
            "sft.weight",
            "UFONE.0.ITLs.0.attn.temperature",
            "UFONE.0.ITLs.0.ffn.project_in.weight",
            "UFONE.0.ITLs.0.ffn.dwconv.weight",
            "UFONE.0.ITLs.0.ffn.project_out.weight",
            "conv_after_body.weight",
            "upsample.0.weight",
        ),
        load=DITN.load,
    ),
    ArchSupport(
        id="MMRealSR",
        detect=_has_keys(
            "conv_first.weight",
            "conv_body.weight",
            "conv_up1.weight",
            "conv_up2.weight",
            "conv_hr.weight",
            "conv_last.weight",
            "body.0.rdb1.conv1.weight",
            "am_list.0.fc.0.weight",
            "de_net.conv_first.0.weight",
            "de_net.body.0.0.conv1.weight",
            "de_net.fc_degree.0.0.weight",
            "dd_embed.0.weight",
        ),
        load=MMRealSR.load,
    ),
    ArchSupport(
        id="SPAN",
        detect=_has_keys(
            "conv_1.sk.weight",
            "block_1.c1_r.sk.weight",
            "block_1.c1_r.eval_conv.weight",
            "block_1.c3_r.eval_conv.weight",
            "conv_cat.weight",
            "conv_2.sk.weight",
            "conv_2.eval_conv.weight",
            "upsampler.0.weight",
        ),
        load=SPAN.load,
    ),
    ArchSupport(
        id="RealCUGAN",
        detect=_has_keys(
            "unet1.conv1.conv.0.weight",
            "unet1.conv1.conv.2.weight",
            "unet1.conv1_down.weight",
            "unet1.conv2.conv.0.weight",
            "unet1.conv2.conv.2.weight",
            "unet1.conv2.seblock.conv1.weight",
            "unet1.conv2_up.weight",
            "unet1.conv_bottom.weight",
            "unet2.conv1.conv.0.weight",
            "unet2.conv1_down.weight",
            "unet2.conv2.conv.0.weight",
            "unet2.conv2.seblock.conv1.weight",
            "unet2.conv3.conv.0.weight",
            "unet2.conv3.seblock.conv1.weight",
            "unet2.conv3_up.weight",
            "unet2.conv4.conv.0.weight",
            "unet2.conv4_up.weight",
            "unet2.conv5.weight",
            "unet2.conv_bottom.weight",
        ),
        load=RealCUGAN.load,
    ),
    ArchSupport(
        id="DDColor",
        detect=_has_keys(
            "mean",
            "std",
            "refine_net.0.0.weight_orig",
            "refine_net.0.0.weight_u",
            "refine_net.0.0.weight_v",
            "encoder.arch.downsample_layers.0.0.weight",
            "encoder.arch.downsample_layers.3.1.weight",
            "encoder.arch.stages.3.0.gamma",
            "encoder.arch.stages.3.0.dwconv.weight",
            "encoder.arch.stages.3.0.pwconv1.weight",
            "encoder.arch.stages.3.0.pwconv2.weight",
            "encoder.arch.norm.weight",
        ),
        load=DDColor.load,
    ),
    ArchSupport(
        id="SAFMN",
        detect=_has_keys(
            "to_feat.weight",
            "feats.0.norm1.weight",
            "feats.0.norm2.weight",
            "feats.0.safm.mfr.0.weight",
            "feats.0.safm.mfr.3.weight",
            "feats.0.safm.aggr.weight",
            "feats.0.ccm.ccm.0.weight",
            "feats.0.ccm.ccm.2.weight",
            "to_img.0.weight",
        ),
        load=SAFMN.load,
    ),
    ArchSupport(
        id="NAFNet",
        detect=_has_keys(
            "intro.weight",
            "ending.weight",
            "ups.0.0.weight",
            "downs.0.weight",
            "middle_blks.0.beta",
            "middle_blks.0.gamma",
            "middle_blks.0.conv1.weight",
            "middle_blks.0.conv2.weight",
            "middle_blks.0.conv3.weight",
            "middle_blks.0.sca.1.weight",
            "middle_blks.0.conv4.weight",
            "middle_blks.0.conv5.weight",
            "middle_blks.0.norm1.weight",
            "middle_blks.0.norm2.weight",
            "encoders.0.0.beta",
            "encoders.0.0.gamma",
            "decoders.0.0.beta",
            "decoders.0.0.gamma",
        ),
        load=NAFNet.load,
    ),
    ArchSupport(
        id="ESRGAN",
        detect=lambda state: (
            _has_keys(
                "model.0.weight",
                "model.1.sub.0.RDB1.conv1.0.weight",
            )(state)
            or _has_keys(
                "conv_first.weight",
                "body.0.rdb1.conv1.weight",
                "conv_body.weight",
                "conv_last.weight",
            )(state)
            # BSRGAN/RealSR
            or _has_keys(
                "conv_first.weight",
                "RRDB_trunk.0.RDB1.conv1.weight",
                "trunk_conv.weight",
                "conv_last.weight",
            )(state)
            # ESRGAN+
            or _has_keys(
                "model.0.weight",
                "model.1.sub.0.RDB1.conv1x1.weight",
            )(state)
        ),
        load=ESRGAN.load,
    ),
)
