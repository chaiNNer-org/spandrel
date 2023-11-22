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

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, TypeVar

from torch import Tensor

# This hack is necessary to make our module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from spandrel import ModelLoader  # noqa: E402

State = Dict[str, object]


def load_state(file: str) -> State:
    return ModelLoader().load_state_dict_from_file(file)


def indent(lines: list[str], indentation: str = "  "):
    def do(line: str) -> str:
        return "\n".join(indentation + s for s in line.splitlines())

    return [do(s) for s in lines]


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
    elif isinstance(v, dict) and all(isinstance(k, str) for k in v.keys()):
        value = f"{get_type_expr(v)} ({str(len(v))})"
        value += "\n" + "\n".join(indent(dump_lines(v)))
    else:
        value = get_type_expr(v)

    k += ": " + " " * (max(0, max_key_len - len(k)))
    return k + value if value else k


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

    by_key: dict[str, State] = {}
    for i, (k, v) in enumerate(state.items()):
        key = current_keys[i]
        if key not in by_key:
            by_key[key] = {}
        key_state = by_key[key]
        key_state[k] = v

    if all([len(v) == 1 for v in by_key.values()]):
        return state

    paths: dict[str, Fork[State] | State] = {}
    for k, v in by_key.items():
        path = group_keys(v, level + 1)

        if isinstance(path, Fork) and len(path.paths) == 1:
            # inline when a fork only has one path
            inner_key = list(path.paths.keys())[0]
            k += "." + inner_key
            path = path.paths[inner_key]

        paths[k] = path

    return Fork(paths)


def dump_lines(state: State) -> list[str]:
    lines: list[str] = []

    def dump(s: Fork[State] | State, level: int = 0):
        indentation = "  " * level

        if isinstance(s, Fork):
            for k, v in s.paths.items():
                lines.append(indentation + k)
                dump(v, level + 1)
        else:
            max_key_len = max([len(k) for k in s.keys()])
            for k, v in s.items():
                lines.append(indentation + get_line(k, v, max_key_len))

    dump(group_keys(state))

    return lines


def dump(state: dict[str, Any], comment: str, file: str = "dump.yml"):
    with open(file, "w") as f:
        comment = "\n".join("# " + s for s in comment.splitlines())
        f.write(f"{comment}\n")
        f.write("\n".join(dump_lines(state)))

    print(f"Dumped {len(state)} keys to {file}")


if __name__ == "__main__":
    file = sys.argv[1]
    print(f"Input file: {file}")
    state = load_state(file)

    dump(state, file)
