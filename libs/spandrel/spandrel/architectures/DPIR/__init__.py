from __future__ import annotations

from typing import Union

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.network_dncnn import IRCNN, DnCNN, FDnCNN

_DPIR = Union[DnCNN, FDnCNN, IRCNN]

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

_is_dncnn = KeyCondition.has_all(
    "model.0.weight",
    "model.0.bias",
    "model.2.weight",
    "model.2.bias",
    KeyCondition.has_any(
        KeyCondition.has_all(
            # act_mode="R"
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
            "model.14.weight",
            "model.14.bias",
        ),
        KeyCondition.has_all(
            # act_mode="BR"
            "model.3.weight",
            "model.3.bias",
            "model.3.running_mean",
            "model.3.running_var",
            "model.5.weight",
            "model.5.bias",
            "model.6.weight",
            "model.6.bias",
            "model.6.running_mean",
            "model.6.running_var",
            "model.8.weight",
            "model.8.bias",
        ),
    ),
)


class DPIRArch(Architecture[_DPIR]):
    def __init__(self) -> None:
        super().__init__(
            id="DPIR",
            detect=KeyCondition.has_any(_is_ircnn, _is_dncnn),
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
        elif _is_dncnn(state_dict):
            # in_nc = 1
            # out_nc = 1
            # nc = 64
            # nb = 17
            # act_mode = "BR"

            in_nc = state_dict["model.0.weight"].shape[1]
            nc = state_dict["model.0.weight"].shape[0]

            layers = get_seq_len(state_dict, "model")
            out_nc = state_dict[f"model.{layers-1}.weight"].shape[0]

            if "model.3.weight" in state_dict:
                act_mode = "BR"
                nb = (layers - 3) // 3 + 2
            else:
                act_mode = "R"
                nb = (layers - 3) // 2 + 2

            tags = ["DnCNN", f"{nc}nc", f"{nb}nb"]
            model = DnCNN(
                in_nc=in_nc,
                out_nc=out_nc,
                nc=nc,
                nb=nb,
                act_mode=act_mode,
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
