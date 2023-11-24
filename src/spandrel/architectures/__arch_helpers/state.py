from __future__ import annotations

import math
from typing import Any


def get_first_seq_index(state: dict, key_pattern: str) -> int:
    """
    Returns the maximum index `i` such that `key_pattern.format(str(i))` is in `state`.

    If no such key is in state, then `-1` is returned.

    Example:
        get_first_seq_index(state, "body.{}.weight") -> -1
        get_first_seq_index(state, "body.{}.weight") -> 3
    """
    for i in range(100):
        if key_pattern.format(str(i)) in state:
            return i
    return -1


def get_seq_len(state: dict[str, Any], seq_key: str) -> int:
    """
    Returns the length of a sequence in the state dict.

    The length is detected by finding the maximum index `i` such that
    `{seq_key}.{i}.{suffix}` is in `state` for some suffix.

    Example:
        get_seq_len(state, "body") -> 5
    """
    prefix = seq_key + "."

    keys: set[int] = set()
    for k in state.keys():
        if k.startswith(prefix):
            index = k[len(prefix) :].split(".", maxsplit=1)[0]
            keys.add(int(index))

    if len(keys) == 0:
        return 0
    return max(keys) + 1


def get_scale_and_output_channels(x: int, input_channels: int) -> tuple[int, int]:
    """
    Returns a scale and number of output channels such that `scale**2 * out_nc = x`.

    This is commonly used for pixelshuffel layers.
    """
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

    # just try out a few candidates and see which ones fulfill the requirements
    candidates = [input_channels, 3, 4, 1]
    for c in candidates:
        if x % c == 0 and is_square(x // c):
            return int(math.sqrt(x // c)), c

    raise AssertionError(
        f"Expected output channels to be either 1, 3, or 4."
        f" Could not find a pair (scale, out_nc) such that `scale**2 * out_nc = {x}`"
    )
