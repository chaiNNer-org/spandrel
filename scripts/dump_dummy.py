# ruff: noqa: E402
"""
This is a tool for dumping the state dict of a dummy model.

Purpose:
When adding/testing model detection or model parameter detection code,
it is useful to see the effects a single parameter has on the state dict of a
model. Since there aren't pretrained models for every possible parameter
configuration, this script can be used to generate a dummy model with the given
parameters.

Usage:
To use this script, you need to edit the `create_dummy` function below. Edit
the function to make it return a model with your desired parameters. As always,
VSCode is the recommended IDE for this task.

After you edited the function, run this script, and it will dump the state dict
of the dummy model to `dump.yml`.

    python scripts/dump_dummy.py

For more detail on the dump itself, see the docs of `dump_state_dict.py`.
"""

import inspect
from textwrap import dedent

import torch
from dump_state_dict import dump
from libs.spandrel.spandrel.architectures import RTMoSR


def create_dummy() -> torch.nn.Module:
    """Edit this function"""
    return RTMoSR.RTMoSR()


if __name__ == "__main__":
    net = create_dummy()
    state = net.state_dict()

    # get source code expression of network
    source = inspect.getsource(create_dummy)
    source = "\n".join(source.split("\n")[2:])  # remove "def create_dummy():
    source = dedent(source)
    if source.startswith("return "):
        source = source[7:]

    dump(state, source)
