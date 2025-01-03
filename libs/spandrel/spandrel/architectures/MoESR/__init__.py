from spandrel.__helpers.size_req import SizeRequirements
from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.MoESR import MoESR, SampleMods


class MoESRArch(Architecture[MoESR]):
    def __init__(self):
        super().__init__(
            id="MoESR",
            # https://github.com/rewaifu/resselt/blob/989dca4fa6d4c32ef09adc948ff26963fee807d5/resselt/archs/moesr/__init__.py#L13
            detect=KeyCondition.has_all(
                "in_to_dim.weight",
                "in_to_dim.bias",
                "blocks.0.blocks.0.gamma",
                "blocks.0.blocks.0.norm.weight",
                "blocks.0.blocks.0.norm.bias",
                "blocks.0.blocks.0.fc1.weight",
                "blocks.0.blocks.0.fc1.bias",
                "blocks.0.blocks.0.conv.dwconv_hw.weight",
                "blocks.0.blocks.0.conv.dwconv_hw.bias",
                "blocks.0.blocks.0.conv.dwconv_w.weight",
                "blocks.0.blocks.0.conv.dwconv_w.bias",
                "blocks.0.blocks.0.conv.dwconv_h.weight",
                "blocks.0.blocks.0.conv.dwconv_h.bias",
                "blocks.0.blocks.0.fc2.weight",
                "blocks.0.blocks.0.fc2.bias",
                "upscale.MetaUpsample",
            ),
        )

    def load(self, state_dict: StateDict) -> ImageModelDescriptor:
        upsample: list[SampleMods] = [
            "conv",
            "pixelshuffledirect",
            "pixelshuffle",
            "nearest+conv",
            "dysample",
        ]
        dim, in_ch = state_dict["in_to_dim.weight"].shape[:2]
        n_blocks = get_seq_len(state_dict, "blocks")
        n_block = get_seq_len(state_dict, "blocks.0.blocks")
        expansion_factor_shape = state_dict["blocks.0.blocks.0.fc1.weight"].shape
        expansion_factor = (expansion_factor_shape[0] / expansion_factor_shape[1]) / 2
        expansion_msg_shape = state_dict["blocks.0.msg.gated.0.fc1.weight"].shape
        expansion_msg = (expansion_msg_shape[0] / expansion_msg_shape[1]) / 2
        _, index, scale, _, out_ch, upsample_dim, _ = state_dict["upscale.MetaUpsample"]
        upsampler = upsample[int(index)]

        model = MoESR(
            in_ch=in_ch,
            out_ch=int(out_ch),
            scale=int(scale),
            n_blocks=n_blocks,
            n_block=n_block,
            dim=dim,
            expansion_factor=expansion_factor,
            expansion_msg=expansion_msg,
            upsampler=upsampler,
            upsample_dim=int(upsample_dim),
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{dim}dim", f"{n_blocks}nbs", f"{n_block}nb", upsampler],
            supports_half=True,
            supports_bfloat16=True,
            scale=int(scale),
            input_channels=in_ch,
            output_channels=int(out_ch),
            size_requirements=SizeRequirements(minimum=2, multiple_of=1, square=False),
        )


__all__ = ["MoESRArch", "MoESR"]
