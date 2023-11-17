from __future__ import annotations

from typing import Callable

from ..architectures import (
    DAT,
    ESRGAN,
    FBCNN,
    GFPGAN,
    HAT,
    MAT,
    SPSR,
    CodeFormer,
    Compact,
    LaMa,
    OmniSR,
    RestoreFormer,
    SCUNet,
    SRFormer,
    SwiftSRGAN,
    Swin2SR,
    SwinIR,
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
        detect=lambda state: (
            "model" in state and "initial.cnn.depthwise.weight" in state["model"].keys()
        ),
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
        id="DAT",
        detect=_has_keys("layers.0.blocks.2.attn.attn_mask_0", "conv_first.weight"),
        load=DAT.load,
    ),
    ArchSupport(
        id="ESRGAN",
        detect=lambda state: (
            _has_keys(
                "model.0.weight",
                "model.1.sub.0.RDB1.conv1.0.weight",
                "model.2.weight",
                "model.4.weight",
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
