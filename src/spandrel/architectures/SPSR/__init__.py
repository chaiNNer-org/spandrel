from ...__helpers.model_descriptor import ImageModelDescriptor, StateDict
from ..__arch_helpers.state import get_seq_len
from .arch.SPSR import SPSRNet as SPSR


def load(state_dict: StateDict) -> ImageModelDescriptor[SPSR]:
    in_nc: int = 3
    out_nc: int = 3
    num_filters: int
    num_blocks: int
    upscale: int = 4
    upsample_mode = "upconv"

    in_nc = state_dict["model.0.weight"].shape[1]
    out_nc = state_dict["f_HR_conv1.0.weight"].shape[0]
    num_filters = state_dict["f_HR_conv1.0.weight"].shape[1]
    num_blocks = get_seq_len(state_dict, "model.1.sub") - 1

    if "model.2.weight" in state_dict:
        upsample_mode = "pixelshuffle"
        upscale_blocks = (get_seq_len(state_dict, "model") - 3) // 3
        upscale = 2**upscale_blocks
    else:
        upsample_mode = "upconv"
        upscale_blocks = (get_seq_len(state_dict, "model") - 3) // 3
        upscale = 2**upscale_blocks

    model = SPSR(
        in_nc=in_nc,
        out_nc=out_nc,
        num_filters=num_filters,
        num_blocks=num_blocks,
        upscale=upscale,
        upsample_mode=upsample_mode,
    )
    tags = [
        f"{num_filters}nf",
        f"{num_blocks}nb",
    ]

    return ImageModelDescriptor(
        model,
        state_dict,
        architecture="SPSR",
        purpose="Restoration" if upscale == 1 else "SR",
        tags=tags,
        supports_half=True,
        supports_bfloat16=True,
        scale=upscale,
        input_channels=in_nc,
        output_channels=out_nc,
    )
