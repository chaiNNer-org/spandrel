import math

from ...__helpers.model_descriptor import SizeRequirements, SRModelDescriptor, StateDict
from .arch.OmniSR import OmniSR


def load(state_dict: StateDict) -> SRModelDescriptor[OmniSR]:
    state = state_dict

    block_num = 1  # Fine to assume this for now
    ffn_bias = True
    pe = True

    num_feat = state_dict["input.weight"].shape[0] or 64
    num_in_ch = state_dict["input.weight"].shape[1] or 3
    num_out_ch = num_in_ch  # we can just assume this for now. pixelshuffle smh

    pixelshuffle_shape = state_dict["up.0.weight"].shape[0]
    up_scale = math.sqrt(pixelshuffle_shape / num_out_ch)
    if up_scale - int(up_scale) > 0:
        print(
            "out_nc is probably different than in_nc, scale calculation might be wrong"
        )
    up_scale = int(up_scale)
    res_num = 0
    for key in state_dict.keys():
        if "residual_layer" in key:
            temp_res_num = int(key.split(".")[1])
            if temp_res_num > res_num:
                res_num = temp_res_num
    res_num = res_num + 1  # zero-indexed

    res_num = res_num

    if (
        "residual_layer.0.residual_layer.0.layer.2.fn.rel_pos_bias.weight"
        in state_dict.keys()
    ):
        rel_pos_bias_weight = state_dict[
            "residual_layer.0.residual_layer.0.layer.2.fn.rel_pos_bias.weight"
        ].shape[0]
        window_size = int((math.sqrt(rel_pos_bias_weight) + 1) / 2)
    else:
        window_size = 8

    model = OmniSR(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        block_num=block_num,
        ffn_bias=ffn_bias,
        pe=pe,
        window_size=window_size,
        res_num=res_num,
        up_scale=up_scale,
        bias=True,
    )

    in_nc = num_in_ch
    out_nc = num_out_ch
    num_feat = num_feat
    scale = up_scale

    tags = [
        f"{num_feat}nf",
        f"w{window_size}",
        f"{res_num}nr",
    ]

    return SRModelDescriptor(
        model,
        state,
        architecture="OmniSR",
        tags=tags,
        supports_half=True,  # TODO: Test this
        supports_bfloat16=True,
        scale=scale,
        input_channels=in_nc,
        output_channels=out_nc,
        size=SizeRequirements(minimum=16),
    )
