from __future__ import annotations

import math
from typing import Union

from typing_extensions import override

from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .arch.cain import CAIN
from .arch.cain_encdec import CAIN_EncDec
from .arch.cain_noca import CAIN_NoCA

_CainModel = Union[CAIN, CAIN_NoCA, CAIN_EncDec]


class CAINArch(Architecture[_CainModel]):
    def __init__(self) -> None:
        super().__init__(
            id="CAIN",
            detect=KeyCondition.has_any(
                # CAIN
                KeyCondition.has_all(
                    "encoder.interpolate.headConv.weight",
                    "encoder.interpolate.headConv.bias",
                    "encoder.interpolate.body.0.body.0.body.0.conv.weight",
                    "encoder.interpolate.body.0.body.0.body.2.conv.weight",
                    "encoder.interpolate.body.0.body.0.body.3.conv_du.0.weight",
                    "encoder.interpolate.body.0.body.1.body.0.conv.weight",
                    "encoder.interpolate.body.0.body.1.body.3.conv_du.2.weight",
                    "encoder.interpolate.body.0.body.3.body.2.conv.weight",
                    "encoder.interpolate.body.3.body.8.body.0.conv.weight",
                    "encoder.interpolate.body.4.body.5.body.3.conv_du.2.weight",
                    "encoder.interpolate.tailConv.weight",
                    "encoder.interpolate.tailConv.bias",
                ),
                # CAIN_NoCA
                KeyCondition.has_all(
                    "encoder.interpolate.headConv.weight",
                    "encoder.interpolate.headConv.bias",
                    "encoder.interpolate.body.0.body.0.body.0.conv.weight",
                    "encoder.interpolate.body.0.body.0.body.2.conv.weight",
                    "encoder.interpolate.body.0.body.1.body.0.conv.weight",
                    "encoder.interpolate.body.0.body.3.body.2.conv.weight",
                    "encoder.interpolate.body.3.body.8.body.0.conv.weight",
                    "encoder.interpolate.tailConv.weight",
                    "encoder.interpolate.tailConv.bias",
                ),
                # CAIN_EncDec
                KeyCondition.has_all(
                    "encoder.body.0.conv.weight",
                    "encoder.body.6.conv.weight",
                    "encoder.interpolate.headConv.weight",
                    "encoder.interpolate.headConv.bias",
                    "encoder.interpolate.body.0.body.0.body.0.conv.weight",
                    "encoder.interpolate.body.0.body.0.body.2.conv.weight",
                    "encoder.interpolate.body.0.body.0.body.3.conv_du.0.weight",
                    "encoder.interpolate.body.0.body.1.body.0.conv.weight",
                    "encoder.interpolate.body.0.body.1.body.3.conv_du.2.weight",
                    "encoder.interpolate.body.0.body.3.body.2.conv.weight",
                    "encoder.interpolate.body.3.body.8.body.0.conv.weight",
                    "encoder.interpolate.body.4.body.5.body.3.conv_du.2.weight",
                    "encoder.interpolate.tailConv.weight",
                    "encoder.interpolate.tailConv.bias",
                    "decoder.body.1.body.0.conv.weight",
                    "decoder.body.1.body.2.conv.bias",
                    "decoder.body.3.body.0.conv.weight",
                    "decoder.body.3.body.2.conv.bias",
                    "decoder.body.5.body.0.conv.weight",
                    "decoder.body.5.body.2.conv.bias",
                ),
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[_CainModel]:
        tags: list[str] = []

        if "encoder.body.0.conv.weight" in state_dict:
            start_filters = 32
            up_mode = "shuffle"

            start_filters = state_dict["decoder.body.5.body.0.conv.weight"].shape[0]

            if "decoder.body.0.upconv.0.conv.weight" in state_dict:
                up_mode = "shuffle"
            elif "decoder.body.0.upconv.weight" in state_dict:
                up_mode = "transpose"
            else:
                up_mode = "direct"

            model = CAIN_EncDec(
                start_filters=start_filters,
                up_mode=up_mode,
            )
            tags.extend(("EncDec", f"{start_filters}sf"))
        else:
            depth = 3

            # detect
            # n_feats = 3 * (4**depth)
            n_feats = state_dict["encoder.interpolate.headConv.weight"].shape[1]
            depth = math.isqrt(math.isqrt(n_feats // 3))

            if (
                "encoder.interpolate.body.0.body.0.body.3.conv_du.0.weight"
                in state_dict
            ):
                model = CAIN(
                    depth=depth,
                )
            else:
                model = CAIN_NoCA(
                    depth=depth,
                )
                tags.append("NoCA")

            tags.append(f"{depth}depth")

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=1,
            input_channels=3,
            output_channels=3,
        )
