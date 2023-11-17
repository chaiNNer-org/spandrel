from ...__helpers.model_descriptor import SRModelDescriptor, StateDict
from .arch.SwiftSRGAN import Generator as SwiftSRGAN


def load(state_dict: StateDict) -> SRModelDescriptor[SwiftSRGAN]:
    state = state_dict
    if "model" in state:
        state = state["model"]

    in_nc: int = state["initial.cnn.depthwise.weight"].shape[0]
    out_nc: int = state["final_conv.pointwise.weight"].shape[0]
    num_filters: int = state["initial.cnn.pointwise.weight"].shape[0]
    num_blocks = len(set([x.split(".")[1] for x in state.keys() if "residual" in x]))
    scale: int = 2 ** len(
        set([x.split(".")[1] for x in state.keys() if "upsampler" in x])
    )

    in_channels = in_nc
    num_channels = num_filters
    num_blocks = num_blocks
    upscale_factor = scale

    model = SwiftSRGAN(
        in_channels=in_channels,
        num_channels=num_channels,
        num_blocks=num_blocks,
        upscale_factor=upscale_factor,
    )
    tags = [
        f"{num_filters}nf",
        f"{num_blocks}nb",
    ]

    return SRModelDescriptor(
        model,
        state,
        architecture="Swift-SRGAN",
        tags=tags,
        supports_half=True,
        supports_bfloat16=True,
        scale=scale,
        input_channels=in_nc,
        output_channels=out_nc,
    )
