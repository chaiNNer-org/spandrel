from ...__helpers.model_descriptor import (
    FaceSRModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.gfpganv1_clean_arch import GFPGANv1Clean


def load(state_dict: StateDict) -> FaceSRModelDescriptor[GFPGANv1Clean]:
    out_size = 512
    num_style_feat = 512
    channel_multiplier = 2
    decoder_load_path = None
    fix_decoder = False
    num_mlp = 8
    input_is_latent = True
    different_w = True
    narrow = 1
    sft_half = True

    model = GFPGANv1Clean(
        out_size=out_size,
        num_style_feat=num_style_feat,
        channel_multiplier=channel_multiplier,
        decoder_load_path=decoder_load_path,
        fix_decoder=fix_decoder,
        num_mlp=num_mlp,
        input_is_latent=input_is_latent,
        different_w=different_w,
        narrow=narrow,
        sft_half=sft_half,
    )

    return FaceSRModelDescriptor(
        model,
        state_dict,
        architecture="GFPGAN",
        tags=[],
        supports_half=False,
        supports_bfloat16=True,
        scale=8,
        input_channels=3,
        output_channels=3,
        size_requirements=SizeRequirements(minimum=512),
    )
