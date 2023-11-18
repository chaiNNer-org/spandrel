import math
import re

from ...__helpers.model_descriptor import SizeRequirements, SRModelDescriptor, StateDict
from .arch.DAT import DAT


def load(state_dict: StateDict) -> SRModelDescriptor[DAT]:
    # defaults
    img_size = 64
    in_chans = 3
    embed_dim = 180
    split_size = [2, 4]
    depth = [2, 2, 2, 2]
    num_heads = [2, 2, 2, 2]
    expansion_factor = 4.0
    upscale = 2
    img_range = 1.0
    resi_connection = "1conv"
    upsampler = "pixelshuffle"

    state_keys = state_dict.keys()
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
        result = re.match(r"layers.(\d*).blocks.(\d*).norm1.weight", key)
        if result:
            layer_num, block_num = result.groups()
            max_layer_num = max(max_layer_num, int(layer_num))
            max_block_num = max(max_block_num, int(block_num))

    depth = [max_block_num + 1 for _ in range(max_layer_num + 1)]

    if "layers.0.blocks.1.attn.temperature" in state_keys:
        num_heads_num = state_dict["layers.0.blocks.1.attn.temperature"].shape[0]
        num_heads = [num_heads_num for _ in range(max_layer_num + 1)]
    else:
        num_heads = depth

    embed_dim = state_dict["conv_first.weight"].shape[0]
    expansion_factor = float(
        state_dict["layers.0.blocks.0.ffn.fc1.weight"].shape[0] / embed_dim
    )

    # TODO: could actually count the layers, but this should do
    if "layers.0.conv.4.weight" in state_keys:
        resi_connection = "3conv"
    else:
        resi_connection = "1conv"

    if "layers.0.blocks.2.attn.attn_mask_0" in state_keys:
        attn_mask_0_x, attn_mask_0_y, _attn_mask_0_z = state_dict[
            "layers.0.blocks.2.attn.attn_mask_0"
        ].shape

        img_size = int(math.sqrt(attn_mask_0_x * attn_mask_0_y))

    if "layers.0.blocks.0.attn.attns.0.rpe_biases" in state_keys:
        split_sizes = state_dict["layers.0.blocks.0.attn.attns.0.rpe_biases"][-1] + 1
        split_size = [int(x) for x in split_sizes]

    in_nc = num_in_ch
    out_nc = num_out_ch
    num_feat = num_feat
    embed_dim = embed_dim
    num_heads = num_heads
    depth = depth
    scale = upscale
    upsampler = upsampler
    img_size = img_size
    img_range = img_range
    expansion_factor = expansion_factor
    resi_connection = resi_connection
    split_size = split_size

    model = DAT(
        img_size=img_size,
        in_chans=in_chans,
        num_feat=num_feat,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        expansion_factor=expansion_factor,
        split_size=split_size,
        scale=scale,
        upsampler=upsampler,
        img_range=img_range,
        resi_connection=resi_connection,
    )

    head_length = len(depth)
    if head_length <= 4:
        size_tag = "small"
    elif head_length < 9:
        size_tag = "medium"
    else:
        size_tag = "large"
    tags = [
        size_tag,
        f"s{img_size}|{split_size[0]}x{split_size[1]}",
        f"{num_feat}nf",
        f"{embed_dim}dim",
        f"{resi_connection}",
    ]

    return SRModelDescriptor(
        model,
        state_dict,
        architecture="DAT",
        tags=tags,
        supports_half=False,  # Too much weirdness to support this at the moment
        supports_bfloat16=True,
        scale=scale,
        input_channels=in_nc,
        output_channels=out_nc,
        size_requirements=SizeRequirements(minimum=16),
    )
