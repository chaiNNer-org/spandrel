from ...__helpers.model_descriptor import ImageModelDescriptor, StateDict
from ..__arch_helpers.state import get_seq_len
from .arch.SwiftSRGAN import Generator as SwiftSRGAN


def load(state: StateDict) -> ImageModelDescriptor[SwiftSRGAN]:
    in_channels: int = 3
    num_channels: int = 64
    num_blocks: int = 16
    upscale_factor: int = 4

    in_channels = state["initial.cnn.depthwise.weight"].shape[0]
    num_channels = state["initial.cnn.pointwise.weight"].shape[0]
    num_blocks = get_seq_len(state, "residual")
    upscale_factor = 2 ** get_seq_len(state, "upsampler")

    model = SwiftSRGAN(
        in_channels=in_channels,
        num_channels=num_channels,
        num_blocks=num_blocks,
        upscale_factor=upscale_factor,
    )
    tags = [
        f"{num_channels}nf",
        f"{num_blocks}nb",
    ]

    return ImageModelDescriptor(
        model,
        state,
        architecture="Swift-SRGAN",
        purpose="Restoration" if upscale_factor == 1 else "SR",
        tags=tags,
        supports_half=True,
        supports_bfloat16=True,
        scale=upscale_factor,
        input_channels=in_channels,
        output_channels=in_channels,
    )
