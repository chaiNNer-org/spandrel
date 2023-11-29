from ...__helpers.model_descriptor import ImageModelDescriptor, StateDict
from ..__arch_helpers.state import get_scale_and_output_channels, get_seq_len
from .arch.SRVGG import SRVGGNetCompact


def load(state_dict: StateDict) -> ImageModelDescriptor[SRVGGNetCompact]:
    state = state_dict

    highest_num = get_seq_len(state, "body") - 1

    in_nc = state["body.0.weight"].shape[1]
    num_feat = state["body.0.weight"].shape[0]
    num_conv = (highest_num - 2) // 2

    pixelshuffle_shape = state[f"body.{highest_num}.bias"].shape[0]
    scale, out_nc = get_scale_and_output_channels(pixelshuffle_shape, in_nc)

    model = SRVGGNetCompact(
        num_in_ch=in_nc,
        num_out_ch=out_nc,
        num_feat=num_feat,
        num_conv=num_conv,
        upscale=scale,
    )

    tags = [f"{num_feat}nf", f"{num_conv}nc"]

    return ImageModelDescriptor(
        model,
        state,
        architecture="RealESRGAN Compact",
        purpose="Restoration" if scale == 1 else "SR",
        tags=tags,
        supports_half=True,
        supports_bfloat16=True,
        scale=scale,
        input_channels=in_nc,
        output_channels=out_nc,
    )
