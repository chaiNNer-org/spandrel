from ...__helpers.model_descriptor import ImageModelDescriptor, StateDict
from ..__arch_helpers.state import get_scale_and_output_channels
from .arch.span import SPAN


def load(state_dict: StateDict) -> ImageModelDescriptor[SPAN]:
    num_in_ch: int = 3
    num_out_ch: int = 3
    feature_channels: int = 48
    upscale: int = 4
    bias = True  # unused internally
    img_range = 255.0  # cannot be deduced from state_dict
    rgb_mean = (0.4488, 0.4371, 0.4040)  # cannot be deduced from state_dict

    num_in_ch = state_dict["conv_1.sk.weight"].shape[1]
    feature_channels = state_dict["conv_1.sk.weight"].shape[0]

    # pixelshuffel shenanigans
    upscale, num_out_ch = get_scale_and_output_channels(
        state_dict["upsampler.0.weight"].shape[0],
        num_in_ch,
    )

    model = SPAN(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        upscale=upscale,
        bias=bias,
        img_range=img_range,
        rgb_mean=rgb_mean,
    )

    return ImageModelDescriptor(
        model,
        state_dict,
        architecture="SPAN",
        purpose="Restoration" if upscale == 1 else "SR",
        tags=[f"{feature_channels}nf"],
        supports_half=True,
        supports_bfloat16=True,
        scale=upscale,
        input_channels=num_in_ch,
        output_channels=num_out_ch,
    )
