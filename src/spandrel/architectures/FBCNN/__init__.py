from ...__helpers.model_descriptor import (
    RestorationModelDescriptor,
    StateDict,
)
from .arch.FBCNN import FBCNN


def load(state_dict: StateDict) -> RestorationModelDescriptor[FBCNN]:
    in_nc = 3
    out_nc = 3
    nc = [64, 128, 256, 512]
    nb = 4
    act_mode = "R"
    downsample_mode = "strideconv"
    upsample_mode = "convtranspose"

    in_nc = state_dict["m_head.weight"].shape[1]
    out_nc = state_dict["m_tail.weight"].shape[0]

    for i in range(0, 20):
        if f"m_down1.{i}.weight" in state_dict:
            nb = i
            break

    nc[0] = state_dict["m_head.weight"].shape[0]
    nc[1] = state_dict[f"m_down1.{nb}.weight"].shape[0]
    nc[2] = state_dict[f"m_down2.{nb}.weight"].shape[0]
    nc[3] = state_dict[f"m_down3.{nb}.weight"].shape[0]

    model = FBCNN(
        in_nc=in_nc,
        out_nc=out_nc,
        nc=nc,
        nb=nb,
        act_mode=act_mode,
        downsample_mode=downsample_mode,
        upsample_mode=upsample_mode,
    )

    return RestorationModelDescriptor(
        model,
        state_dict,
        architecture="FBCNN",
        tags=[],
        supports_half=True,  # TODO
        supports_bfloat16=True,  # TODO
        input_channels=in_nc,
        output_channels=out_nc,
    )
