from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.usrnet import USRNet


class USRNetArch(Architecture[USRNet]):
    def __init__(self) -> None:
        super().__init__(
            id="USRNet",
            detect=KeyCondition.has_all(
                "p.m_head.weight",
                "p.m_down1.0.res.0.weight",
                "p.m_down1.0.res.2.weight",
                "p.m_down2.0.res.0.weight",
                "p.m_down2.0.res.2.weight",
                "p.m_down3.0.res.0.weight",
                "p.m_down3.0.res.2.weight",
                "p.m_body.0.res.0.weight",
                "p.m_body.0.res.2.weight",
                "p.m_tail.weight",
                "h.mlp.0.weight",
                "h.mlp.0.bias",
                "h.mlp.2.weight",
                "h.mlp.2.bias",
                "h.mlp.4.weight",
                "h.mlp.4.bias",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[USRNet]:
        # n_iter = 8
        # h_nc = 64
        # in_nc = 4
        # out_nc = 3
        # nc = [64, 128, 256, 512]
        # nb = 2
        act_mode = "R"
        # downsample_mode = "strideconv"
        # upsample_mode = "convtranspose"

        # detect parameters
        n_iter = state_dict["h.mlp.4.weight"].shape[0] // 2
        h_nc = state_dict["h.mlp.0.weight"].shape[0]

        in_nc = state_dict["p.m_head.weight"].shape[1]
        out_nc = state_dict["p.m_tail.weight"].shape[0]

        nc = [
            state_dict["p.m_down1.0.res.0.weight"].shape[0],
            state_dict["p.m_down2.0.res.0.weight"].shape[0],
            state_dict["p.m_down3.0.res.0.weight"].shape[0],
            state_dict["p.m_body.0.res.0.weight"].shape[0],
        ]
        nb = get_seq_len(state_dict, "p.m_body")

        if f"p.m_down1.{nb}.weight" in state_dict:
            downsample_mode = "strideconv"
        else:
            # we cannot distinguish between avgpool and maxpool
            downsample_mode = "maxpool"

        if "p.m_up3.1.res.0.weight" in state_dict:
            upsample_mode = "convtranspose"
        elif "p.m_up3.0.weight" in state_dict:
            upsample_mode = "pixelshuffle"
        elif "p.m_up3.1.weight" in state_dict:
            upsample_mode = "upconv"
        else:
            raise ValueError("Unknown upsample mode")

        model = USRNet(
            n_iter=n_iter,
            h_nc=h_nc,
            in_nc=in_nc,
            out_nc=out_nc,
            nc=nc,
            nb=nb,
            act_mode=act_mode,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[],
            supports_half=False,  # TODO: check
            supports_bfloat16=True,
            scale=1,
            input_channels=in_nc,
            output_channels=out_nc,
            size_requirements=SizeRequirements(multiple_of=128, square=True),
        )
