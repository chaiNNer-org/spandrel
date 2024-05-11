from __future__ import annotations

from typing_extensions import override

from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.network_ludvae import LUDVAE


class LUDVAEArch(Architecture[LUDVAE]):
    def __init__(self) -> None:
        super().__init__(
            id="LUDVAE",
            detect=KeyCondition.has_all(
                "inconv.weight",
                "inconv.bias",
                "inconv_n.bias",
                "enc_1.net.0.weight",
                "enc_1.net.6.bias",
                "enc_2.net.0.weight",
                "enc_2.net.6.bias",
                "enc_3.net.0.weight",
                "enc_3.net.6.bias",
                "enc_n_1.net.0.weight",
                "enc_n_1.net.6.bias",
                "enc_n_2.net.0.weight",
                "enc_n_2.net.6.bias",
                "enc_n_3.net.0.weight",
                "enc_n_3.net.6.bias",
                "Gauconv_q_3.m.weight",
                "Gauconv_q_3.v.weight",
                "Gauconv_q_2.m.weight",
                "Gauconv_q_2.v.weight",
                "Gauconv_q_1.m.weight",
                "Gauconv_q_1.v.weight",
                "Gauconv_p_2.m.weight",
                "Gauconv_p_2.v.weight",
                "Gauconv_p_1.m.weight",
                "Gauconv_p_1.v.weight",
                "dec_3.net.0.weight",
                "dec_3.net.6.bias",
                "dec_2.net.0.weight",
                "dec_2.net.6.bias",
                "dec_1.net.0.weight",
                "dec_1.net.6.bias",
                "proj_3.proj.weight",
                "proj_2.proj.weight",
                "proj_c_2.proj.weight",
                "proj_n_2.proj.weight",
                "proj_1.proj.weight",
                "proj_c_1.proj.weight",
                "proj_n_1.proj.weight",
                "outconv.weight",
                "outconv.bias",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[LUDVAE]:
        in_channel = 3
        filters_num = 128

        in_channel = state_dict["inconv.weight"].shape[1]
        filters_num = state_dict["inconv.weight"].shape[0]

        model = LUDVAE(
            in_channel=in_channel,
            filters_num=filters_num,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[f"{filters_num}nf"],
            supports_half=False,  # TODO: Test this
            supports_bfloat16=True,
            scale=1,
            input_channels=in_channel,
            output_channels=in_channel,
            size_requirements=SizeRequirements(),
        )
