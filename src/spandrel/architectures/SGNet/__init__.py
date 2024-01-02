from ...__helpers.model_descriptor import GuidedImageModelDescriptor, StateDict
from .arch.sgnet import SGNet


def _migrate(state_dict: StateDict):
    # this fixes up some renamed keys
    # see https://github.com/yanzq95/SGNet/issues/6 for more details
    for key in list(state_dict.keys()):
        if ".panprocess." in key:
            new_key = key.replace(".panprocess.", ".rgbprocess.")
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        if ".panpre." in key:
            new_key = key.replace(".panpre.", ".rgbpre.")
            state_dict[new_key] = state_dict[key]
            del state_dict[key]


def load(state_dict: StateDict) -> GuidedImageModelDescriptor[SGNet]:
    _migrate(state_dict)

    # this arch doesn't have default values
    num_feats: int
    kernel_size: int
    scale: int

    num_feats = state_dict["conv_rgb1.weight"].shape[0]
    kernel_size = state_dict["conv_rgb1.weight"].shape[2]
    scale = state_dict["upsampler.conv_1.0.weight"].shape[2] - 4

    model = SGNet(
        num_feats=num_feats,
        kernel_size=kernel_size,
        scale=scale,
    )

    return GuidedImageModelDescriptor(
        model,
        state_dict,
        architecture="SGNet",
        purpose="GuidedSR",
        tags=[f"{num_feats}nf"],
        supports_half=False,  # TODO: check this
        supports_bfloat16=True,
        scale=scale,
        input_channels=1,
        output_channels=1,
        guide_input_channels=3,
        call_fn=lambda model, x, guide: model((guide, x))[0],
    )
