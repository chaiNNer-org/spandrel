import math
import re

from torch import nn

from ...__helpers.model_descriptor import SizeRequirements, SRModelDescriptor, StateDict
from .arch.SRFormer import SRFormer


def load(state_dict: StateDict) -> SRModelDescriptor[SRFormer]:
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
    qk_scale = None
    drop_rate = 0.0
    attn_drop_rate = 0.0
    drop_path_rate = 0.1
    norm_layer = nn.LayerNorm
    ape = False
    patch_norm = True
    upscale = 2
    img_range = 1.0
    upsampler = ""
    resi_connection = "1conv"

    state = state_dict

    state_keys = list(state_dict.keys())

    if "conv_before_upsample.0.weight" in state_keys:
        if "conv_up1.weight" in state_keys:
            upsampler = "nearest+conv"
        else:
            upsampler = "pixelshuffle"
    elif "upsample.0.weight" in state_keys:
        upsampler = "pixelshuffledirect"
    else:
        upsampler = ""

    num_feat_pre_layer = state_dict.get("conv_before_upsample.weight", None)
    num_feat_layer = state_dict.get("conv_before_upsample.0.weight", None)
    num_feat = (
        num_feat_layer.shape[1]
        if num_feat_layer is not None and num_feat_pre_layer is not None
        else 64
    )

    num_in_ch = state_dict["conv_first.weight"].shape[1]
    in_chans = num_in_ch
    if "conv_last.weight" in state_keys:
        num_out_ch = state_dict["conv_last.weight"].shape[0]
    else:
        num_out_ch = num_in_ch

    upscale = 1
    if upsampler == "nearest+conv":
        upsample_keys = [x for x in state_keys if "conv_up" in x and "bias" not in x]

        for upsample_key in upsample_keys:
            upscale *= 2
    elif upsampler == "pixelshuffle":
        upsample_keys = [
            x
            for x in state_keys
            if "upsample" in x and "conv" not in x and "bias" not in x
        ]
        for upsample_key in upsample_keys:
            shape = state_dict[upsample_key].shape[0]
            upscale *= math.sqrt(shape // num_feat)
        upscale = int(upscale)
    elif upsampler == "pixelshuffledirect":
        upscale = int(math.sqrt(state_dict["upsample.0.bias"].shape[0] // num_out_ch))

    max_layer_num = 0
    max_block_num = 0
    for key in state_keys:
        result = re.match(r"layers.(\d*).residual_group.blocks.(\d*).norm1.weight", key)
        if result:
            layer_num, block_num = result.groups()
            max_layer_num = max(max_layer_num, int(layer_num))
            max_block_num = max(max_block_num, int(block_num))

    depths = [max_block_num + 1 for _ in range(max_layer_num + 1)]

    if (
        "layers.0.residual_group.blocks.0.attn.relative_position_bias_table"
        in state_keys
    ):
        num_heads_num = state_dict[
            "layers.0.residual_group.blocks.0.attn.relative_position_bias_table"
        ].shape[-1]
        num_heads = [num_heads_num for _ in range(max_layer_num + 1)]
    else:
        num_heads = depths

    embed_dim = state_dict["conv_first.weight"].shape[0]

    mlp_ratio = float(
        state_dict["layers.0.residual_group.blocks.0.mlp.fc1.bias"].shape[0] / embed_dim
    )

    # TODO: could actually count the layers, but this should do
    # TOOD: confirm this is correct and the same as SwinIR
    if "layers.0.conv.4.weight" in state_keys:
        resi_connection = "3conv"
    else:
        resi_connection = "1conv"

    window_size = int(
        math.sqrt(
            state_dict[
                "layers.0.residual_group.blocks.0.attn.aligned_relative_position_index"
            ].shape[0]
        )
    )

    if "layers.0.residual_group.blocks.1.attn_mask" in state_keys:
        img_size = int(
            (
                math.sqrt(
                    state_dict["layers.0.residual_group.blocks.1.attn_mask"].shape[0]
                )
            )
            * 16
        )

    in_nc = num_in_ch
    out_nc = num_out_ch
    num_feat = num_feat
    embed_dim = embed_dim
    num_heads = num_heads
    depths = depths
    window_size = window_size
    mlp_ratio = mlp_ratio
    scale = upscale
    upsampler = upsampler
    img_size = img_size
    img_range = img_range
    resi_connection = resi_connection

    model = SRFormer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_feat=num_feat,
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
        norm_layer=norm_layer,
        ape=ape,
        patch_norm=patch_norm,
        upscale=scale,
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
        f"{num_feat}nf",
        f"{embed_dim}dim",
        f"{resi_connection}",
    ]

    return SRModelDescriptor(
        model,
        state,
        architecture="SRFormer",
        tags=tags,
        supports_half=False,  # Too much weirdness to support this at the moment
        supports_bfloat16=True,
        scale=scale,
        input_channels=in_nc,
        output_channels=out_nc,
        size_requirements=SizeRequirements(minimum=16),
    )
