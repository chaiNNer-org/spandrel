from __future__ import annotations

import math
from typing import Literal

from ...__helpers.canonicalize import remove_common_prefix
from ...__helpers.model_descriptor import SRModelDescriptor, StateDict
from ..__arch_helpers.state import get_scale_and_output_channels, get_seq_len
from .arch.grl import GRL as GRLIR


def _clean_up_checkpoint(state_dict: StateDict) -> StateDict:
    # The official checkpoints are all over the place.

    # Issue 1: some models prefix all keys with "model."
    state_dict = remove_common_prefix(state_dict, ["model."])

    # Issue 2: some models have a bunch of useless keys and prefix all important keys with "model_g."
    # (looking at you, `bsr_grl_base.ckpt`)
    if "model_g.conv_first.weight" in state_dict:
        # only keep keys with "model_g." prefix
        state_dict = {k: v for k, v in state_dict.items() if k.startswith("model_g.")}
        state_dict = remove_common_prefix(state_dict, ["model_g."])

    return state_dict


def _get_output_params(state_dict: StateDict, in_channels: int):
    out_channels: int
    upsampler: str
    upscale: int

    num_out_feats = 64  # hard-coded
    if (
        "conv_before_upsample.0.weight" in state_dict
        and "upsample.up.0.weight" in state_dict
    ):
        upsampler = "pixelshuffle"
        out_channels = state_dict["conv_last.weight"].shape[0]

        upscale = 1
        for i in range(0, get_seq_len(state_dict, "upsample.up"), 2):
            shape = state_dict[f"upsample.up.{i}.weight"].shape[0]
            upscale *= int(math.sqrt(shape // num_out_feats))
    elif "upsample.up.0.weight" in state_dict:
        upsampler = "pixelshuffledirect"
        upscale, out_channels = get_scale_and_output_channels(
            state_dict["upsample.up.0.weight"].shape[0], in_channels
        )
    elif "conv_up1.weight" in state_dict:
        upsampler = "nearest+conv"
        out_channels = state_dict["conv_last.weight"].shape[0]
        upscale = 4  # only supports 4x
    else:
        upsampler = ""
        out_channels = state_dict["conv_last.weight"].shape[0]
        upscale = 1

    return out_channels, upsampler, upscale


def _get_anchor_params(
    state_dict: StateDict, default_down_factor: int
) -> tuple[bool, str, int]:
    anchor_one_stage: bool
    anchor_proj_type: str
    anchor_window_down_factor: int

    anchor_body_len = get_seq_len(state_dict, "layers.0.blocks.0.attn.anchor.body")
    if anchor_body_len == 1:
        anchor_one_stage = True

        if "layers.0.blocks.0.attn.anchor.body.0.reduction.weight" in state_dict:
            if "layers.0.blocks.0.attn.anchor.body.0.reduction.bias" in state_dict:
                # We can deduce neither proj_type nor window_down_factor.
                # So we'll just assume the values the official configs use.
                anchor_proj_type = "avgpool"  # or "maxpool", who knows?
                anchor_window_down_factor = default_down_factor
            else:
                anchor_proj_type = "patchmerging"
                # window_down_factor is undefined here
                anchor_window_down_factor = default_down_factor
        elif "layers.0.blocks.0.attn.anchor.body.0.weight" in state_dict:
            anchor_proj_type = "conv2d"
            anchor_window_down_factor = (
                state_dict["layers.0.blocks.0.attn.anchor.body.0.weight"].shape[2] - 1
            )
        else:
            anchor_proj_type = "separable_conv"
            anchor_window_down_factor = (
                state_dict["layers.0.blocks.0.attn.anchor.body.0.0.weight"].shape[2] - 1
            )
    else:
        anchor_one_stage = False
        anchor_window_down_factor = 2**anchor_body_len

        if "layers.0.blocks.0.attn.anchor.body.0.reduction.weight" in state_dict:
            anchor_proj_type = "patchmerging"
        elif "layers.0.blocks.0.attn.anchor.body.0.weight" in state_dict:
            anchor_proj_type = "conv2d"
        else:
            anchor_proj_type = "separable_conv"

    return anchor_one_stage, anchor_proj_type, anchor_window_down_factor


def load(state_dict: StateDict) -> SRModelDescriptor[GRLIR]:
    state_dict = _clean_up_checkpoint(state_dict)

    img_size: int = 64
    # in_channels: int = 3
    # out_channels: int = 3
    # embed_dim:int = 96
    # upscale:int = 2
    # upsampler = ""
    # depths: list[int] = [6, 6, 6, 6, 6, 6]
    # num_heads_window: list[int] = [3, 3, 3, 3, 3, 3]
    # num_heads_stripe: list[int] = [3, 3, 3, 3, 3, 3]
    window_size: int = 8  # cannot be deduced from state_dict
    stripe_size: list[int] = [8, 8]  # cannot be deduced from state_dict
    stripe_groups: list[int | None] = [None, None]  # cannot be deduced from state_dict
    stripe_shift: bool = False  # cannot be deduced from state_dict
    # mlp_ratio: float = 4.0
    # qkv_bias: bool = True
    # qkv_proj_type: Literal["linear", "separable_conv"] = "linear"
    # anchor_proj_type = "avgpool"
    # anchor_one_stage: bool = True
    # anchor_window_down_factor: int = 1
    out_proj_type: Literal["linear", "conv2d"] = "linear"  # unused internally
    # local_connection: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    pretrained_window_size: list[int] = [0, 0]  # cannot be deduced from state_dict
    pretrained_stripe_size: list[int] = [0, 0]  # cannot be deduced from state_dict
    # conv_type = "1conv"
    init_method = "n"  # cannot be deduced from state_dict
    euclidean_dist: bool = False  # cannot be deduced from state_dict

    in_channels = state_dict["conv_first.weight"].shape[1]
    embed_dim = state_dict["conv_first.weight"].shape[0]

    out_channels, upsampler, upscale = _get_output_params(state_dict, in_channels)

    # conv_type
    if "conv_after_body.weight" in state_dict:
        conv_after_body_shape = state_dict["conv_after_body.weight"].shape
        if len(conv_after_body_shape) == 2:
            conv_type = "linear"
        elif conv_after_body_shape[2] == 1:
            conv_type = "1conv1x1"
        else:
            conv_type = "1conv"
    else:
        conv_type = "3conv"

    # depths
    depths_len = get_seq_len(state_dict, "layers")
    depths = [6] * depths_len
    num_heads_window = [3] * depths_len
    num_heads_stripe = [3] * depths_len
    for i in range(depths_len):
        depths[i] = get_seq_len(state_dict, f"layers.{i}.blocks")
        num_heads_window[i] = state_dict[
            f"layers.{i}.blocks.0.attn.window_attn.attn_transform.logit_scale"
        ].shape[0]
        num_heads_stripe[i] = state_dict[
            f"layers.{i}.blocks.0.attn.stripe_attn.attn_transform1.logit_scale"
        ].shape[0]

    # qkv
    if "layers.0.blocks.0.attn.qkv.body.weight" in state_dict:
        qkv_proj_type = "linear"
        qkv_bias = "layers.0.blocks.0.attn.qkv.body.bias" in state_dict
    else:
        qkv_proj_type = "separable_conv"
        qkv_bias = "layers.0.blocks.0.attn.qkv.body.0.bias" in state_dict

    # anchor
    anchor_one_stage, anchor_proj_type, anchor_window_down_factor = _get_anchor_params(
        state_dict,
        # We use 4 as the default value (if the value cannot be detected), because
        # that's what all the official models use.
        default_down_factor=4,
    )

    # other
    local_connection = "layers.0.blocks.0.conv.cab.0.weight" in state_dict
    mlp_ratio = state_dict["layers.0.blocks.0.mlp.fc1.weight"].shape[0] / embed_dim

    # Set undetectable parameters.
    # These parameters are huge pain, because they vary widely between models, so we'll
    # just use some heuristics to support the official models, and call it a day.
    if upscale == 1:
        # denoise (dn), deblur (db), demosaic (dm), or jpeg
        pass
    else:
        # sr or bsr
        if upsampler == "nearest+conv":
            # bsr
            window_size = 16
            stripe_size = [32, 64]
        else:
            # sr
            window_size = 32
            stripe_size = [64, 64]

    model = GRLIR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        upscale=upscale,
        upsampler=upsampler,
        depths=depths,
        num_heads_window=num_heads_window,
        num_heads_stripe=num_heads_stripe,
        window_size=window_size,
        stripe_size=stripe_size,
        stripe_groups=stripe_groups,
        stripe_shift=stripe_shift,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qkv_proj_type=qkv_proj_type,
        anchor_proj_type=anchor_proj_type,
        anchor_one_stage=anchor_one_stage,
        anchor_window_down_factor=anchor_window_down_factor,
        out_proj_type=out_proj_type,
        local_connection=local_connection,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        pretrained_window_size=pretrained_window_size,
        pretrained_stripe_size=pretrained_stripe_size,
        conv_type=conv_type,
        init_method=init_method,
        euclidean_dist=euclidean_dist,
    )

    size_tag = "base"
    if len(depths) < 6:
        size_tag = "small" if embed_dim >= 96 else "tiny"

    return SRModelDescriptor(
        model,
        state_dict,
        architecture="GRLIR",
        tags=[
            size_tag,
            f"{embed_dim}dim",
            f"w{window_size}df{anchor_window_down_factor}",
            f"s{stripe_size[0]}x{stripe_size[1]}",
        ],
        supports_half=False,
        supports_bfloat16=True,
        scale=upscale,
        input_channels=in_channels,
        output_channels=out_channels,
    )
