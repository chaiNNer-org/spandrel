from __future__ import annotations

from typing import Literal, Union

import torch
from typing_extensions import override

from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.upcunet_v3 import UpCunet2x, UpCunet3x, UpCunet4x

_RealCUGAN = Union[UpCunet2x, UpCunet3x, UpCunet4x]


class RealCUGANArch(Architecture[_RealCUGAN]):
    def __init__(self) -> None:
        super().__init__(
            id="RealCUGAN",
            license="MIT",
            detect=KeyCondition.has_all(
                "unet1.conv1.conv.0.weight",
                "unet1.conv1.conv.2.weight",
                "unet1.conv1_down.weight",
                "unet1.conv2.conv.0.weight",
                "unet1.conv2.conv.2.weight",
                "unet1.conv2.seblock.conv1.weight",
                "unet1.conv2_up.weight",
                "unet1.conv_bottom.weight",
                "unet2.conv1.conv.0.weight",
                "unet2.conv1_down.weight",
                "unet2.conv2.conv.0.weight",
                "unet2.conv2.seblock.conv1.weight",
                "unet2.conv3.conv.0.weight",
                "unet2.conv3.seblock.conv1.weight",
                "unet2.conv3_up.weight",
                "unet2.conv4.conv.0.weight",
                "unet2.conv4_up.weight",
                "unet2.conv5.weight",
                "unet2.conv_bottom.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[_RealCUGAN]:
        scale: Literal[2, 3, 4]
        in_channels: int
        out_channels: int
        pro: bool = False

        tags: list[str] = []
        if "pro" in state_dict:
            pro = True
            tags.append("pro")
            state_dict["pro"] = torch.zeros(1)

        if "conv_final.weight" in state_dict:
            # UpCunet4x
            scale = 4
            in_channels = state_dict["unet1.conv1.conv.0.weight"].shape[1]
            out_channels = 3  # hard coded in UpCunet4x
            model = UpCunet4x(
                in_channels=in_channels, out_channels=out_channels, pro=pro
            )
        elif state_dict["unet1.conv_bottom.weight"].shape[2] == 5:
            # UpCunet3x
            scale = 3
            in_channels = state_dict["unet1.conv1.conv.0.weight"].shape[1]
            out_channels = state_dict["unet2.conv_bottom.weight"].shape[0]
            model = UpCunet3x(
                in_channels=in_channels, out_channels=out_channels, pro=pro
            )
        else:
            # UpCunet2x
            scale = 2
            in_channels = state_dict["unet1.conv1.conv.0.weight"].shape[1]
            out_channels = state_dict["unet2.conv_bottom.weight"].shape[0]
            model = UpCunet2x(
                in_channels=in_channels, out_channels=out_channels, pro=pro
            )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="SR",
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_channels,
            output_channels=out_channels,
            size_requirements=SizeRequirements(minimum=32),
        )
