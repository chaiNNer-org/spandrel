from __future__ import annotations

from ...__helpers.model_descriptor import (
    RestorationModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.kbnet_l import KBNet_l
from .arch.kbnet_s import KBNet_s

# KBCNN is essentially 2 similar but different architectures: KBNet_l and KBNet_s.


def _get_max_seq(state: StateDict, key_pattern: str, start: int = 0) -> int:
    """
    Returns the maximum number `i` such that `key_pattern.format(str(i))` is in `state`.

    If no such key is in state, then `start - 1` is returned.
    """
    i = start
    while True:
        key = key_pattern.format(str(i))
        if key not in state:
            return i - 1
        i += 1


def load_l(state_dict: StateDict) -> RestorationModelDescriptor[KBNet_l]:
    in_nc = 3
    out_nc = 3
    dim = 48
    num_blocks = [4, 6, 6, 8]
    num_refinement_blocks = 4
    heads = [1, 2, 4, 8]
    ffn_expansion_factor = 1.5
    bias = False

    in_nc = state_dict["patch_embed.proj.weight"].shape[1]
    out_nc = state_dict["output.weight"].shape[0]

    dim = state_dict["patch_embed.proj.weight"].shape[0]

    num_blocks[0] = _get_max_seq(state_dict, "encoder_level1.{}.norm1.weight") + 1
    num_blocks[1] = _get_max_seq(state_dict, "encoder_level2.{}.norm1.weight") + 1
    num_blocks[2] = _get_max_seq(state_dict, "encoder_level3.{}.norm1.weight") + 1
    num_blocks[3] = _get_max_seq(state_dict, "latent.{}.norm1.weight") + 1

    num_refinement_blocks = _get_max_seq(state_dict, "refinement.{}.norm1.weight") + 1

    heads[0] = state_dict["encoder_level1.0.ffn.temperature"].shape[0]
    heads[1] = state_dict["encoder_level2.0.ffn.temperature"].shape[0]
    heads[2] = state_dict["encoder_level3.0.ffn.temperature"].shape[0]
    heads[3] = state_dict["latent.0.ffn.temperature"].shape[0]

    bias = "encoder_level1.0.ffn.qkv.bias" in state_dict

    # in code: hidden_features = int(dim * ffn_expansion_factor)
    hidden_features = state_dict["encoder_level1.0.attn.ga1"].shape[1]
    ffn_expansion_factor = hidden_features / dim

    model = KBNet_l(
        inp_channels=in_nc,
        out_channels=out_nc,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        heads=heads,
        ffn_expansion_factor=ffn_expansion_factor,
        bias=bias,
    )

    return RestorationModelDescriptor(
        model,
        state_dict,
        architecture="KBCNN",
        tags=["L"],
        supports_half=True,  # TODO
        supports_bfloat16=True,  # TODO
        input_channels=in_nc,
        output_channels=out_nc,
        size_requirements=SizeRequirements(multiple_of=16),
    )


def load_s(state_dict: StateDict) -> RestorationModelDescriptor[KBNet_s]:
    img_channel = 3
    width = 64
    middle_blk_num = 12
    enc_blk_nums = [2, 2, 4, 8]
    dec_blk_nums = [2, 2, 2, 2]
    lightweight = False
    ffn_scale = 2

    img_channel = state_dict["intro.weight"].shape[1]
    width = state_dict["intro.weight"].shape[0]

    middle_blk_num = _get_max_seq(state_dict, "middle_blks.{}.w") + 1

    enc_count = _get_max_seq(state_dict, "encoders.{}.0.w") + 1
    enc_blk_nums = [1] * enc_count
    for i in range(enc_count):
        enc_blk_nums[i] = _get_max_seq(state_dict, "encoders." + str(i) + ".{}.w") + 1

    dec_count = _get_max_seq(state_dict, "decoders.{}.0.w") + 1
    dec_blk_nums = [1] * dec_count
    for i in range(dec_count):
        dec_blk_nums[i] = _get_max_seq(state_dict, "decoders." + str(i) + ".{}.w") + 1

    # in code: ffn_ch = int(c * ffn_scale)
    temp_c = state_dict["middle_blks.0.conv4.weight"].shape[1]
    temp_ffn_ch = state_dict["middle_blks.0.conv4.weight"].shape[0]
    ffn_scale = temp_ffn_ch / temp_c

    model = KBNet_s(
        img_channel=img_channel,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums,
        lightweight=lightweight,
        ffn_scale=ffn_scale,
    )

    return RestorationModelDescriptor(
        model,
        state_dict,
        architecture="KBCNN",
        tags=["S"],
        supports_half=True,  # TODO
        supports_bfloat16=True,  # TODO
        input_channels=img_channel,
        output_channels=img_channel,
    )


def load(
    state_dict: StateDict
) -> RestorationModelDescriptor[KBNet_l] | RestorationModelDescriptor[KBNet_s]:
    if "patch_embed.proj.weight" in state_dict:
        return load_l(state_dict)
    else:
        return load_s(state_dict)
