from __future__ import annotations

import math

from ...__helpers.model_descriptor import SRModelDescriptor, StateDict
from ..__arch_helpers.state import get_max_seq_index
from .arch.SRVGG import SRVGGNetCompact


def _get_scale_and_output_channels(x: int, input_channels: int) -> tuple[int, int]:
    # Unfortunately, we do not have enough information to determine both the scale and
    # number output channels correctly *in general*. However, we can make some
    # assumptions to make it good enough.
    #
    # What we know:
    # - x = scale * scale * output_channels
    # - output_channels is likely equal to input_channels
    # - output_channels and input_channels is likely 1, 3, or 4
    # - scale is likely 1, 2, 4, or 8

    def is_square(n: int) -> bool:
        return math.sqrt(n) == int(math.sqrt(n))

    def perfect_square_root(n: int) -> int:
        root = int(math.sqrt(n))
        assert root * root == n, f"{n} is not a perfect square"
        return root

    if x % 3 == 0 and x % 9 != 0:
        # we know that output_channels MUST be a multiple of 3
        # so let's assume that output_channels is exactly 3
        x = x // 3
        return perfect_square_root(x), 3

    # just try out a few candidates and see which ones fulfill the requirements
    candidates = [input_channels, 3, 4, 1]
    for c in candidates:
        if x % c == 0 and is_square(x // c):
            return perfect_square_root(x // c), c

    raise AssertionError(
        f"Expected output channels to be either 1, 3, or 4. Could not find a a pair (s, o) such that s*s*o = {x}"
    )


def load(state_dict: StateDict) -> SRModelDescriptor[SRVGGNetCompact]:
    state = state_dict

    highest_num = get_max_seq_index(state, "body.{}.weight")

    in_nc = state["body.0.weight"].shape[1]
    num_feat = state["body.0.weight"].shape[0]
    num_conv = (highest_num - 2) // 2

    pixelshuffle_shape = state[f"body.{highest_num}.bias"].shape[0]
    scale, out_nc = _get_scale_and_output_channels(pixelshuffle_shape, in_nc)

    model = SRVGGNetCompact(
        num_in_ch=in_nc,
        num_out_ch=out_nc,
        num_feat=num_feat,
        num_conv=num_conv,
        upscale=scale,
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
