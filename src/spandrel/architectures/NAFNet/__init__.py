from __future__ import annotations

from ...__helpers.model_descriptor import (
    ImageModelDescriptor,
    StateDict,
)
from ..__arch_helpers.state import get_seq_len
from .arch.NAFNet_arch import NAFNet


def load(state_dict: StateDict) -> ImageModelDescriptor[NAFNet]:
    # default values
    img_channel: int = 3
    width: int = 16
    middle_blk_num: int = 1
    enc_blk_nums: list[int] = []
    dec_blk_nums: list[int] = []

    img_channel = state_dict["intro.weight"].shape[1]
    width = state_dict["intro.weight"].shape[0]
    middle_blk_num = get_seq_len(state_dict, "middle_blks")
    for i in range(get_seq_len(state_dict, "encoders")):
        enc_blk_nums.append(get_seq_len(state_dict, f"encoders.{i}"))
    for i in range(get_seq_len(state_dict, "decoders")):
        dec_blk_nums.append(get_seq_len(state_dict, f"decoders.{i}"))

    model = NAFNet(
        img_channel=img_channel,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums,
    )

    return ImageModelDescriptor(
        model,
        state_dict,
        architecture="NAFNet",
        purpose="Restoration",
        tags=[f"{width}w"],
        supports_half=False,  # TODO: Test this
        supports_bfloat16=True,
        scale=1,
        input_channels=img_channel,
        output_channels=img_channel,
    )
