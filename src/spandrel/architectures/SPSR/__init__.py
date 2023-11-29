from ...__helpers.model_descriptor import ImageModelDescriptor, StateDict
from .arch.SPSR import SPSRNet as SPSR


def get_scale(state: StateDict, min_part: int = 4) -> int:
    n = 0
    for part in list(state):
        parts = part.split(".")
        if len(parts) == 3:
            part_num = int(parts[1])
            if part_num > min_part and parts[0] == "model" and parts[2] == "weight":
                n += 1
    return 2**n


def get_num_blocks(state: StateDict) -> int:
    nb = 0
    for part in list(state):
        parts = part.split(".")
        n_parts = len(parts)
        if n_parts == 5 and parts[2] == "sub":
            nb = int(parts[3])
    return nb


def load(state_dict: StateDict) -> ImageModelDescriptor[SPSR]:
    state = state_dict

    num_blocks = get_num_blocks(state)

    in_nc: int = state["model.0.weight"].shape[1]
    out_nc: int = state["f_HR_conv1.0.bias"].shape[0]

    scale = get_scale(state, 4)
    num_filters: int = state["model.0.weight"].shape[0]

    model = SPSR(
        in_nc=in_nc,
        out_nc=out_nc,
        num_filters=num_filters,
        num_blocks=num_blocks,
        upscale=scale,
    )
    tags = [
        f"{num_filters}nf",
        f"{num_blocks}nb",
    ]

    return ImageModelDescriptor(
        model,
        state,
        architecture="SPSR",
        purpose="Restoration" if scale == 1 else "SR",
        tags=tags,
        supports_half=True,
        supports_bfloat16=True,
        scale=scale,
        input_channels=in_nc,
        output_channels=out_nc,
    )
