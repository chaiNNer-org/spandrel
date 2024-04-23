from __future__ import annotations

from typing import Literal

from typing_extensions import override

from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .arch.flavr import JoinType
from .arch.flavr import UNet_3D_3D as FLAVR


class FLAVRArch(Architecture[FLAVR]):
    def __init__(self) -> None:
        super().__init__(
            id="FLAVR",
            detect=KeyCondition.has_all(
                "encoder.stem.0.weight",
                "encoder.layer1.0.conv1.0.weight",
                "encoder.layer1.0.conv2.0.weight",
                "encoder.layer1.0.fg.attn_layer.0.weight",
                "encoder.layer1.1.conv1.0.weight",
                "encoder.layer4.0.conv1.0.weight",
                "decoder.0.conv.0.weight",
                "decoder.0.conv.0.bias",
                "decoder.0.conv.1.attn_layer.0.weight",
                "decoder.0.conv.1.attn_layer.0.bias",
                "feature_fuse.conv.0.weight",
                "outconv.1.weight",
                "outconv.1.bias",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[FLAVR]:
        block: Literal["unet_18", "unet_34"] = "unet_18"
        n_inputs: int = 4
        n_outputs: int = 3
        batchnorm = False
        join_type: JoinType = "concat"
        upmode = "transpose"

        block = (
            "unet_34" if "encoder.layer1.2.conv1.0.weight" in state_dict else "unet_18"
        )
        n_inputs = state_dict["feature_fuse.conv.0.weight"].shape[1] // 64
        n_outputs = state_dict["outconv.1.weight"].shape[0] // 3
        batchnorm = "feature_fuse.conv.1.weight" in state_dict

        if "decoder.1.upconv.0.weight" in state_dict:
            upmode = "transpose"
            growth = state_dict["decoder.1.upconv.0.weight"].shape[0] // 256
        else:
            upmode = "direct"
            growth = state_dict["decoder.1.upconv.1.weight"].shape[1] // 256

        if growth == 2:
            join_type = "concat"
        else:
            # we can't differentiate between add and first
            join_type = "add"

        model = FLAVR(
            block=block,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            batchnorm=batchnorm,
            joinType=join_type,
            upmode=upmode,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[],
            supports_half=True,
            supports_bfloat16=True,
            scale=1,
            input_channels=3,
            output_channels=3,
        )
