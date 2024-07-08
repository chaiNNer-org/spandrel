from __future__ import annotations

from typing import Literal

from typing_extensions import override

from spandrel import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from spandrel.util import KeyCondition, get_seq_len

from .__arch.mirnet_v2_arch import MIRNet_v2 as MIRNet2


class MIRNet2Arch(Architecture[MIRNet2]):
    def __init__(self) -> None:
        super().__init__(
            id="MIRNet2",
            detect=KeyCondition.has_all(
                "conv_in.weight",
                "body.0.body.0.dau_top.body.0.weight",
                "body.0.body.0.dau_top.body.2.weight",
                "body.0.body.0.dau_top.gcnet.conv_mask.weight",
                "body.0.body.0.dau_top.gcnet.channel_add_conv.0.weight",
                "body.0.body.0.dau_top.gcnet.channel_add_conv.2.weight",
                "body.0.body.0.dau_mid.body.0.weight",
                "body.0.body.0.dau_mid.gcnet.channel_add_conv.0.weight",
                "body.0.body.0.dau_bot.body.0.weight",
                "body.0.body.0.dau_bot.gcnet.channel_add_conv.0.weight",
                "body.0.body.0.down2.body.0.bot.1.weight",
                "body.0.body.0.down4.0.body.0.bot.1.weight",
                "body.0.body.0.down4.1.body.0.bot.1.weight",
                "body.0.body.0.up21_1.body.0.bot.0.weight",
                "body.0.body.0.up21_2.body.0.bot.0.weight",
                "body.0.body.0.up32_1.body.0.bot.0.weight",
                "body.0.body.0.up32_2.body.0.bot.0.weight",
                "body.0.body.0.conv_out.weight",
                "body.0.body.0.skff_top.conv_du.0.weight",
                "body.0.body.0.skff_top.fcs.0.weight",
                "body.0.body.0.skff_mid.conv_du.0.weight",
                "body.0.body.0.skff_mid.fcs.0.weight",
                "body.1.body.0.dau_top.body.0.weight",
                "body.2.body.0.dau_top.body.0.weight",
                "body.3.body.0.dau_top.body.0.weight",
                "conv_out.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[MIRNet2]:
        # inp_channels = 3
        # out_channels = 3
        # n_feat = 80
        # chan_factor = 1.5
        # n_MRB = 2
        # bias = False
        task: Literal["defocus_deblurring"] | None = None

        inp_channels = state_dict["conv_in.weight"].shape[1]
        out_channels = state_dict["conv_out.weight"].shape[0]
        n_feat = state_dict["conv_in.weight"].shape[0]
        bias = "conv_in.bias" in state_dict

        n_MRB = get_seq_len(state_dict, "body.0.body") - 1  # noqa: N806
        chan_factor = (
            state_dict["body.0.body.0.dau_mid.body.0.weight"].shape[0] / n_feat
        )

        if inp_channels == 6:
            task = "defocus_deblurring"

        model = MIRNet2(
            inp_channels=inp_channels,
            out_channels=out_channels,
            n_feat=n_feat,
            chan_factor=chan_factor,
            n_MRB=n_MRB,
            bias=bias,
            task=task,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[f"{n_feat}nf"],
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=1,
            input_channels=inp_channels,
            output_channels=out_channels,
            size_requirements=SizeRequirements(multiple_of=4),
        )


__all__ = ["MIRNet2Arch", "MIRNet2"]
