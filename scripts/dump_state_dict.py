"""
This is a tool for dumping the state dict of a model to a human-readable format.

The output is a YAML file called `dump.yml`. This file contains all keys of the
state dict grouped by common prefixes. This makes it easier to analyze the
structure of the state dict.

The format is also very good for diffing. To find out the differences between 2
similar models do the following:

1. Dump the state dicts of model 1.
2. Use git to stage `dump.yml`.
3. Dump the state dicts of model 2.
4. `git diff dump.yml` or use an IDE to see the differences.

Usage:

    python scripts/dump_state_dict.py /path/to/model.pth

Example `dump.yml`:

    # model.pth
    model
      0
        model.0.weight: Tensor float32 Size([64, 3, 3, 3])
        model.0.bias:   Tensor float32 Size([64])
      1.sub
        0
          RDB1
            conv1.0
              model.1.sub.0.RDB1.conv1.0.weight: Tensor float32 Size([32, 64, 3, 3])
              model.1.sub.0.RDB1.conv1.0.bias:   Tensor float32 Size([32])
            conv2.0
              model.1.sub.0.RDB1.conv2.0.weight: Tensor float32 Size([32, 96, 3, 3])
              model.1.sub.0.RDB1.conv2.0.bias:   Tensor float32 Size([32])

`dump.yml` should ideally be opened in an IDE that supports YAML syntax and
code folding. These features make it easier to understand the structure of the
state dict.
"""

from __future__ import annotations

import argparse
import collections
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, Mapping, TypeVar

from torch import Tensor



State = Dict[str, object]


def get_iter_type_union_expr(collection: Iterable[object]) -> str:
    types = {get_type_expr(i) for i in collection}
    if len(types) == 0:
        return "Never"
    return " | ".join(types)


def get_collection_type_expr(collection: dict | set | list | tuple) -> str:
    if isinstance(collection, dict):
        key = get_iter_type_union_expr(collection.keys())
        value = get_iter_type_union_expr(collection.values())
        return f"{type(collection).__name__}[{key}, {value}]"

    return f"{type(collection).__name__}[{get_iter_type_union_expr(collection)}]"


def get_type_expr(x: object) -> str:
    if isinstance(x, (dict, set, list, tuple)):
        return get_collection_type_expr(x)
    return type(x).__name__


def get_line(k: str, v: object, max_key_len: int = 0) -> str:
    value: str
    if isinstance(v, (int, float, str, bool)):
        value = str(v)
    elif isinstance(v, Tensor):
        dtype = str(v.dtype).replace("torch.", "")
        shape = str(v.shape).replace("torch.", "")
        value = f"Tensor {dtype} {shape}"
    elif isinstance(v, dict) and all(isinstance(k, str) for k in v):
        value = f"{get_type_expr(v)} ({len(v)})"
        value += "\n" + textwrap.indent("\n".join(dump_lines(group_keys(v))), "  ")
    else:
        value = get_type_expr(v)

    if value:
        k += ":"
        return f"{k.ljust(max_key_len + 2)}{value}"
    return k


T = TypeVar("T")


@dataclass
class Fork(Generic[T]):
    paths: dict[str, Fork[T] | T]


def group_keys(state: State, level: int = 0) -> Fork[State] | State:
    if len(state) <= 1:
        return state

    current_keys: list[str] = []
    for k in state.keys():
        parts = k.split(".")
        if len(parts) <= level:
            return state
        current_keys.append(parts[level])

    by_key: Mapping[str, State] = collections.defaultdict(dict)
    for key, (k, v) in zip(current_keys, state.items()):
        by_key[key][k] = v

    if all(len(v) == 1 for v in by_key.values()):
        return state

    paths: dict[str, Fork[State] | State] = {}
    for k, v in by_key.items():
        path = group_keys(v, level + 1)

        if isinstance(path, Fork) and len(path.paths) == 1:
            # inline when a fork only has one path
            inner_key = next(iter(path.paths))
            k += "." + inner_key
            path = path.paths[inner_key]

        paths[k] = path

    return Fork(paths)


def dump_lines(s: Fork[State] | State, *, level: int = 0):
    indentation = "  " * level

    if isinstance(s, Fork):
        for k, v in s.paths.items():
            yield indentation + k
            yield from dump_lines(v, level=level + 1)
    else:
        max_key_len = max(len(k) for k in s)
        for k, v in s.items():
            yield indentation + get_line(k, v, max_key_len)


def dump(state: dict[str, Any], comment: str, file: str = "dump.yml"):
    with open(file, "w") as f:
        for line in comment.splitlines():
            print("#", line, file=f)
        for line in dump_lines(group_keys(state)):
            print(line, file=f)

    print(f"Dumped {len(state)} keys to {file}")

import torch
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="Path to model file")
    ap.add_argument("-o", "--output", help="Output file", default="dump.yml")
    args = ap.parse_args()
    file = args.file
    print(f"Input file: {file}")
    state = torch.load(file, map_location='cpu')
    dump(state, comment=file, file=args.output)


if __name__ == "__main__":
    main()
