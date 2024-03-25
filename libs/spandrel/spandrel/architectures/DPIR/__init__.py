from __future__ import annotations

from typing import Union

from typing_extensions import override

from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.network_dncnn import IRCNN, FDnCNN

_DPIR = Union[FDnCNN, IRCNN]

_is_ircnn = KeyCondition.exactly(
    "model.0.weight",
    "model.0.bias",
    "model.2.weight",
    "model.2.bias",
    "model.4.weight",
    "model.4.bias",
    "model.6.weight",
    "model.6.bias",
    "model.8.weight",
    "model.8.bias",
    "model.10.weight",
    "model.10.bias",
    "model.12.weight",
    "model.12.bias",
)


class DPIRArch(Architecture[_DPIR]):
    def __init__(self) -> None:
        super().__init__(
            id="DPIR",
            detect=KeyCondition.has_any(_is_ircnn),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[_DPIR]:
        if _is_ircnn(state_dict):
            # in_nc = 1
            # out_nc = 1
            # nc = 64

            in_nc = state_dict["model.0.weight"].shape[1]
            out_nc = state_dict["model.12.weight"].shape[0]
            nc = state_dict["model.0.weight"].shape[0]

            tags = ["IRCNN", f"{nc}nc"]
            model = IRCNN(
                in_nc=in_nc,
                out_nc=out_nc,
                nc=nc,
            )
        else:
            raise ValueError("Invalid state dict")

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=tags,
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=1,
            input_channels=in_nc,
            output_channels=out_nc,
            size_requirements=SizeRequirements(),
        )
