import math

from ...__helpers.model_descriptor import (
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from ...architectures.__arch_helpers.state import get_seq_len
from .arch.SRFormer import SRFormer


def load(state_dict: StateDict) -> ImageModelDescriptor[SRFormer]:
    # Default
    img_size = 64
    patch_size = 1
    in_chans = 3
    embed_dim = 96
    depths = (6, 6, 6, 6)
    num_heads = (6, 6, 6, 6)
    window_size = 7
    mlp_ratio = 4.0
    qkv_bias = True
    qk_scale = None  # cannot be deduced from state_dict
    drop_rate = 0.0  # cannot be deduced from state_dict
    attn_drop_rate = 0.0  # cannot be deduced from state_dict
    drop_path_rate = 0.1  # cannot be deduced from state_dict
    ape = False
    patch_norm = True
    upscale = 2
    img_range = 1.0
    upsampler = ""
    resi_connection = "1conv"

    in_chans = state_dict["conv_first.weight"].shape[1]
    embed_dim = state_dict["conv_first.weight"].shape[0]

    ape = "absolute_pos_embed" in state_dict
    patch_norm = "patch_embed.norm.weight" in state_dict
    qkv_bias = "layers.0.residual_group.blocks.0.attn.q.bias" in state_dict

    mlp_ratio = float(
        state_dict["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0]
        / embed_dim
    )

    # depths & num_heads
    num_layers = get_seq_len(state_dict, "layers")
    depths = [6] * num_layers
    num_heads = [6] * num_layers
    for i in range(num_layers):
        depths[i] = get_seq_len(state_dict, f"layers.{i}.residual_group.blocks")
        num_heads[i] = state_dict[
            f"layers.{i}.residual_group.blocks.0.attn.relative_position_bias_table"
        ].shape[1]

    if "conv_hr.weight" in state_dict:
        upsampler = "nearest+conv"
        upscale = 4  # only supported scale
    elif "conv_before_upsample.0.weight" in state_dict:
        upsampler = "pixelshuffle"

        num_feat = 64  # hard-coded constant
        upscale = 1
        for i in range(0, get_seq_len(state_dict, "upsample"), 2):
            shape = state_dict[f"upsample.{i}.weight"].shape[0]
            upscale *= int(math.sqrt(shape // num_feat))
    elif "upsample.0.weight" in state_dict:
        upsampler = "pixelshuffledirect"
        upscale = int(math.sqrt(state_dict["upsample.0.weight"].shape[0] // in_chans))
    else:
        upsampler = ""
        upscale = 1  # it's technically undefined, but we'll use 1

    if "conv_after_body.weight" in state_dict:
        resi_connection = "1conv"
    else:
        resi_connection = "3conv"

    window_size = (
        int(
            math.sqrt(
                state_dict[
                    "layers.0.residual_group.blocks.0.attn.relative_position_bias_table"
                ].shape[0]
            )
        )
        + 1
    )

    # Unfortunately, we cannot detect img_size and patch_size, but we can detect
    # patches_resolution. What we know:
    #   patches_resolution = img_size // patch_size
    #   if window_size > patches_resolution:
    #     attn_mask[0] = patches_resolution**2 // window_size**2
    # We will assume that we already know the patch_size (we don't, we'll assume the default value).
    if "layers.0.residual_group.blocks.1.attn_mask" in state_dict:
        attn_mask_0 = state_dict["layers.0.residual_group.blocks.1.attn_mask"].shape[0]
        patches_resolution = int(math.sqrt(attn_mask_0 * window_size * window_size))
    else:
        # we only know that window_size <= patches_resolution
        # assume window_size == patches_resolution
        patches_resolution = window_size

        # if APE is enabled, we know that absolute_pos_embed[1] == patches_resolution**2
        if ape:
            patches_resolution = int(math.sqrt(state_dict["absolute_pos_embed"][1]))
    img_size = patch_size * patches_resolution
    # Further, img_size is actually rounded up to the nearest multiple of window_size
    # before calculating patches_resolution. We have to do a bit of guess to get
    # the actual img_size...
    for nice_number in [512, 256, 128, 96, 64, 48, 32, 24, 16]:
        rounded = nice_number
        if rounded % window_size != 0:
            rounded = rounded + (window_size - rounded % window_size)
        if rounded // patch_size == patches_resolution:
            img_size = nice_number
            break

    model = SRFormer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        upscale=upscale,
        upsampler=upsampler,
        img_range=img_range,
        resi_connection=resi_connection,
    )

    head_length = len(depths)
    if head_length <= 4:
        size_tag = "small"
    elif head_length < 9:
        size_tag = "medium"
    else:
        size_tag = "large"
    tags = [
        size_tag,
        f"s{img_size}w{window_size}",
        f"{embed_dim}dim",
        f"{resi_connection}",
    ]

    return ImageModelDescriptor(
        model,
        state_dict,
        architecture="SRFormer",
        purpose="Restoration" if upscale == 1 else "SR",
        tags=tags,
        supports_half=False,  # Too much weirdness to support this at the moment
        supports_bfloat16=True,
        scale=upscale,
        input_channels=in_chans,
        output_channels=in_chans,
        size_requirements=SizeRequirements(minimum=16),
    )
