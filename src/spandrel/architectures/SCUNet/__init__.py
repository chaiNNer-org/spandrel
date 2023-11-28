from ...__helpers.model_descriptor import (
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from ..__arch_helpers.state import get_seq_len
from .arch.SCUNet import SCUNet


def load(state_dict: StateDict) -> ImageModelDescriptor[SCUNet]:
    in_nc = 3
    config = [4, 4, 4, 4, 4, 4, 4]
    dim = 64
    drop_path_rate = 0.0
    input_resolution = 256

    dim = state_dict["m_head.0.weight"].shape[0]
    in_nc = state_dict["m_head.0.weight"].shape[1]

    config[0] = get_seq_len(state_dict, "m_down1") - 1
    config[1] = get_seq_len(state_dict, "m_down2") - 1
    config[2] = get_seq_len(state_dict, "m_down3") - 1
    config[3] = get_seq_len(state_dict, "m_body")
    config[4] = get_seq_len(state_dict, "m_up3") - 1
    config[5] = get_seq_len(state_dict, "m_up2") - 1
    config[6] = get_seq_len(state_dict, "m_up1") - 1

    model = SCUNet(
        in_nc=in_nc,
        config=config,
        dim=dim,
        drop_path_rate=drop_path_rate,
        input_resolution=input_resolution,
    )

    return ImageModelDescriptor(
        model,
        state_dict,
        architecture="SCUNet",
        purpose="Restoration",
        tags=[],
        supports_half=True,
        supports_bfloat16=True,
        scale=1,
        input_channels=in_nc,
        output_channels=in_nc,
        size_requirements=SizeRequirements(minimum=16),
    )
