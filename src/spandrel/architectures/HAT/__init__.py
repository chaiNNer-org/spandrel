import math
import re

from torch import nn

from ...__helpers.model_descriptor import SizeRequirements, SRModelDescriptor, StateDict
from .arch.HAT import HAT


def load(state_dict: StateDict) -> SRModelDescriptor[HAT]:
    # Defaults
    img_size = 64
    patch_size = 1
    in_chans = 3
    embed_dim = 96
    depths = (6, 6, 6, 6)
    num_heads = (6, 6, 6, 6)
    window_size = 7
    overlap_ratio = 0.5
    mlp_ratio = 4.0
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.0
    attn_drop_rate = 0.0
    drop_path_rate = 0.1
    norm_layer = nn.LayerNorm
    ape = False
    patch_norm = True
    use_checkpoint = False
    upscale = 2
    img_range = 1.0
    upsampler = ""
    resi_connection = "1conv"

    state_keys = list(state_dict.keys())
    state = state_dict

    num_feat = state_dict["conv_last.weight"].shape[1]
    in_chans = state_dict["conv_first.weight"].shape[1]
    num_out_ch = state_dict["conv_last.weight"].shape[0]
    embed_dim = state_dict["conv_first.weight"].shape[0]

    if "conv_before_upsample.0.weight" in state_keys:
        if "conv_up1.weight" in state_keys:
            upsampler = "nearest+conv"
        else:
            upsampler = "pixelshuffle"
    elif "upsample.0.weight" in state_keys:
        upsampler = "pixelshuffledirect"
    else:
        upsampler = ""
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
            shape = state[upsample_key].shape[0]
            upscale *= math.sqrt(shape // num_feat)
        upscale = int(upscale)
    elif upsampler == "pixelshuffledirect":
        upscale = int(math.sqrt(state["upsample.0.bias"].shape[0] // num_out_ch))

    max_layer_num = 0
    max_block_num = 0
    for key in state_keys:
        result = re.match(
            r"layers.(\d*).residual_group.blocks.(\d*).conv_block.cab.0.weight", key
        )
        if result:
            layer_num, block_num = result.groups()
            max_layer_num = max(max_layer_num, int(layer_num))
            max_block_num = max(max_block_num, int(block_num))

    depths = [max_block_num + 1 for _ in range(max_layer_num + 1)]

    if (
        "layers.0.residual_group.blocks.0.attn.relative_position_bias_table"
        in state_keys
    ):
        num_heads_num = state[
            "layers.0.residual_group.blocks.0.attn.relative_position_bias_table"
        ].shape[-1]
        num_heads = [num_heads_num for _ in range(max_layer_num + 1)]
    else:
        num_heads = depths

    mlp_ratio = float(
        state["layers.0.residual_group.blocks.0.mlp.fc1.bias"].shape[0] / embed_dim
    )

    # TODO: could actually count the layers, but this should do
    if "layers.0.conv.4.weight" in state_keys:
        resi_connection = "3conv"
    else:
        resi_connection = "1conv"

    window_size = int(math.sqrt(state["relative_position_index_SA"].shape[0]))

    # Not sure if this is needed or used at all anywhere in HAT's config
    if "layers.0.residual_group.blocks.1.attn_mask" in state_keys:
        img_size = int(
            math.sqrt(state["layers.0.residual_group.blocks.1.attn_mask"].shape[0])
            * window_size
        )

    window_size = window_size
    shift_size = window_size // 2
    overlap_ratio = overlap_ratio

    in_nc = in_chans
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

    model = HAT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_nc,
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
        use_checkpoint=use_checkpoint,
        upscale=upscale,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
        num_feat=num_feat,
        num_out_ch=out_nc,
        shift_size=shift_size,
    )

    head_length = len(depths)  # type: ignore
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
        state_dict,
        architecture="HAT",
        tags=tags,
        supports_half=False,
        supports_bfloat16=True,
        scale=scale,
        input_channels=in_nc,
        output_channels=out_nc,
        size=SizeRequirements(minimum=16),
    )
