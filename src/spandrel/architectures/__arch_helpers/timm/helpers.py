""" Layer/Module Helpers
Hacked together by / Copyright 2020 Ross Wightman
"""
from __future__ import annotations

import collections.abc
from itertools import repeat
from typing import Iterable, TypeVar


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
# to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

T = TypeVar("T")


def to_2tuple(x: T | Iterable[T]) -> tuple[T, T]:
    if isinstance(x, str):
        return x, x  # type: ignore
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)  # type: ignore
    return x, x


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v
