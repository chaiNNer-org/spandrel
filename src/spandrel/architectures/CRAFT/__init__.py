import re
from ..__arch_helpers.state import get_scale_and_output_channels
from ...__helpers.model_descriptor import (
    ImageModelDescriptor,
    SizeRequirements,
    StateDict
)

from .arch.CRAFT import CRAFT

def load(state_dict: StateDict) -> ImageModelDescriptor[CRAFT]:
    # Required values
    in_chans = state_dict['conv_first.weight'].shape[1]
    embed_dim = state_dict['layers.0.conv.weight'].shape[1]

    depths = []
    num_heads = []

    split_size_0 = 4
    split_size_1 = 16

    mlp_ratio = 2.

    qkv_bias = True
    qk_scale = None

    upscale = int(math.sqrt(state_dict['upsample.0.bias'].shape[0] / in_chans))
    img_range = 1.
    resi_connection = '1conv'

    for key, tensor in state_dict.items():
        depth_match = re.search(r'layers.(\d+).residual_group.srwa_blocks.(\d+).norm1.weight', key)
        if depth_match and int(depth_match.group(2)) % 2 == 0:
            layer = int(depth_match.group(1))
            if len(depths) - 1 < layer:
                depths.append(0)

            depths[layer] += 1
            continue
        
        if re.fullmatch(r'layers.\d+.residual_group.hf_blocks.0.attn.temperature', key):
            num_heads.append(tensor.shape[0])

    # Tag values
    num_feat = state_dict['upsample.0.weight'].shape[1]

    model = CRAFT(
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        split_size_0=split_size_0,
        split_size_1=split_size_1,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        upscale=upscale,
        img_range=img_range,
        resi_connection=resi_connection
    )

    tags = [
        f'{split_size_0}x{split_size_1}',
        f'{num_feat}nf',
        f'{embed_dim}dim',
        f'{resi_connection}'
    ]

    return ImageModelDescriptor(
        model,
        state_dict,
        architecture="CRAFT",
        purpose="Restoration" if upscale == 1 else "SR",
        tags=tags,
        supports_half=True, # TODO: Not thoroughly tested
        supports_bfloat16=True,
        scale=upscale,
        input_channels=in_chans,
        output_channels=in_chans,
        size_requirements=SizeRequirements(minimum=16, multiple_of=16)
    )
