from typing_extensions import override

from spandrel import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from spandrel.util import KeyCondition, get_seq_len

from .__arch.M3SNet import M3SNet


class M3SNetArch(Architecture[M3SNet]):
    def __init__(self) -> None:
        super().__init__(
            id="M3SNet",
            detect=KeyCondition.has_all(
                "intro.weight",
                "intro.bias",
                "ending.weight",
                "ending.bias",
                "encoders.0.0.beta",
                "encoders.0.0.gamma",
                "encoders.0.0.conv1.weight",
                "encoders.0.0.conv2.weight",
                "encoders.0.0.conv3.weight",
                "encoders.0.0.sca.1.weight",
                "encoders.0.0.conv4.weight",
                "encoders.0.0.conv5.weight",
                "encoders.0.0.norm1.weight",
                "encoders.0.0.norm2.weight",
                "decoders.0.0.conv1.weight",
                "decoders.0.0.conv2.weight",
                "decoders.0.0.conv3.weight",
                "decoders.0.0.sca.1.weight",
                "decoders.0.0.conv4.weight",
                "decoders.0.0.conv5.weight",
                "decoders.0.0.norm1.weight",
                "decoders.0.0.norm2.weight",
                "middle_blks.0.temperature",
                "middle_blks.0.qkv.weight",
                "middle_blks.0.qkv_dwconv.weight",
                "middle_blks.0.project_out.weight",
                "ups.0.0.weight",
                "downs.0.weight",
                "middle.0.0.conv1.weight",
                "middle.0.0.conv2.weight",
                "middle.0.0.conv3.weight",
                "middle.0.0.sca.1.weight",
                "middle.0.0.conv4.weight",
                "middle.0.0.conv5.weight",
                "middle.0.0.norm1.weight",
                "middle.0.0.norm2.weight",
                "middle_ups.0.0.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[M3SNet]:
        img_channel = 3
        width = 32
        middle_blk_num = 1
        enc_blk_nums = [1, 1, 1, 28]
        dec_blk_nums = [1, 1, 1, 1]

        img_channel = state_dict["intro.weight"].shape[1]
        width = state_dict["intro.weight"].shape[0]
        middle_blk_num = get_seq_len(state_dict, "middle_blks")

        enc_blk_nums = []
        for i in range(get_seq_len(state_dict, "encoders")):
            enc_blk_nums.append(get_seq_len(state_dict, f"encoders.{i}"))

        dec_blk_nums = []
        for i in range(get_seq_len(state_dict, "decoders")):
            dec_blk_nums.append(get_seq_len(state_dict, f"decoders.{i}"))

        model = M3SNet(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[f"{width}w"],
            supports_half=False,  # TODO: check if this is true
            supports_bfloat16=True,
            scale=1,
            input_channels=img_channel,
            output_channels=img_channel,
            size_requirements=SizeRequirements(multiple_of=16),
        )


__all__ = ["M3SNetArch", "M3SNet"]
