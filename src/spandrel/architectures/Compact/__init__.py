from __future__ import annotations

import math

from ...__helpers.model_descriptor import SRModelDescriptor, StateDict
from .arch.SRVGG import SRVGGNetCompact


def _get_num_conv(highest_num: int) -> int:
    return (highest_num - 2) // 2


def _get_num_feats(state: StateDict, weight_keys: list[str]) -> int:
    return state[weight_keys[0]].shape[0]


def _get_in_nc(state: StateDict, weight_keys: list[str]) -> int:
    return state[weight_keys[0]].shape[1]


def _get_scale(pixelshuffle_shape: int, out_nc: int) -> int:
    scale = math.sqrt(pixelshuffle_shape / out_nc)
    if scale - int(scale) > 0:
        print(
            "out_nc is probably different than in_nc, scale calculation might be wrong"
        )
    scale = int(scale)
    return scale


def load(state_dict: StateDict) -> SRModelDescriptor[SRVGGNetCompact]:
    state = state_dict

    weight_keys = [key for key in state.keys() if "weight" in key]
    highest_num = max([int(key.split(".")[1]) for key in weight_keys if "body" in key])

    in_nc = _get_in_nc(state, weight_keys)
    num_feat = _get_num_feats(state, weight_keys)
    num_conv = _get_num_conv(highest_num)
    # Assume out_nc is the same as in_nc
    # I cant think of a better way to do that
    out_nc = in_nc  # :(

    pixelshuffle_shape = state[f"body.{highest_num}.bias"].shape[0]
    scale = _get_scale(pixelshuffle_shape, out_nc)

    model = SRVGGNetCompact(
        num_in_ch=in_nc,
        num_out_ch=out_nc,
        num_feat=num_feat,
        num_conv=num_conv,
        upscale=scale,
        pixelshuffle_shape=pixelshuffle_shape,
    )

    tags = [f"{num_feat}nf", f"{num_conv}nc"]

    return SRModelDescriptor(
        model,
        state,
        architecture="RealESRGAN Compact",
        tags=tags,
        supports_half=True,
        supports_bfloat16=True,
        scale=scale,
        input_channels=in_nc,
        output_channels=out_nc,
    )
