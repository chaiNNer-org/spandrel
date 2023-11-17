from ...__helpers.model_descriptor import (
    RestorationModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.SCUNet import SCUNet


def load(state_dict: StateDict) -> RestorationModelDescriptor[SCUNet]:
    in_nc = 3
    config = [4, 4, 4, 4, 4, 4, 4]
    dim = 64
    drop_path_rate = 0.0
    input_resolution = 256

    dim = state_dict["m_head.0.weight"].shape[0]
    in_nc = state_dict["m_head.0.weight"].shape[1]

    model = SCUNet(
        in_nc=in_nc,
        config=config,
        dim=dim,
        drop_path_rate=drop_path_rate,
        input_resolution=input_resolution,
    )

    return RestorationModelDescriptor(
        model,
        state_dict,
        architecture="SCUNet",
        tags=[],
        supports_half=True,
        supports_bfloat16=True,
        input_channels=in_nc,
        output_channels=in_nc,
        size=SizeRequirements(minimum=16),
    )
