import math

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.lhan import LHAN


class LHANArch(Architecture[LHAN]):
    def __init__(self):
        super().__init__(
            id="LHAN",
            detect=KeyCondition.has_all(
                "groups.0.blocks.0.n1.weight",
            ),
        )

    def load(self, state_dict: StateDict) -> ImageModelDescriptor[LHAN]:
        num_in_ch = state_dict["conv_first.weight"].shape[1]
        num_out_ch = num_in_ch  # TODO
        upscaling_factor = 4
        embed_dim = state_dict["conv_first.weight"].shape[0]
        num_groups = get_seq_len(state_dict, "groups")
        group_block_pattern = ["spatial", "channel"]
        depth_per_group = get_seq_len(state_dict, "groups.0.blocks") // len(
            group_block_pattern
        )
        num_heads = state_dict["groups.0.blocks.0.attn.bias"].shape[0]
        window_size = math.isqrt(state_dict["groups.0.blocks.0.attn.bias"].shape[2])
        ffn_expansion_ratio = float(
            state_dict["groups.0.blocks.0.ffn.fc1.weight"].shape[0] / embed_dim
        )
        aim_reduction_ratio = (
            embed_dim // state_dict["groups.0.blocks.0.inter.cg.1.weight"]
        )

        drop_path_rate = 0

        upsampler_type = "pixelshuffle"
        if "upsampler.conv.0.weight" in state_dict:
            upsampler_type = "nearest_conv"
            # TODO upscaling factor
        elif "upsampler.refine.weight" in state_dict:
            upsampler_type = "transpose_conv"
            if "upsampler.up.0.weight" in state_dict:
                upscaling_factor = 4
            elif state_dict["upsampler.up.weight"].shape[-1] == 3:
                upscaling_factor = 3
            else:
                upscaling_factor = 2
        elif "upsampler.conv_pre.weight" in state_dict:
            upsampler_type = "pixelshuffle"
            upscaling_factor = int(
                math.sqrt(state_dict["upsampler.conv_pre.bias"].shape[0] // num_out_ch)
            )

        img_range = 1.0
        model = LHAN(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            upscaling_factor=upscaling_factor,
            embed_dim=embed_dim,
            num_groups=num_groups,
            depth_per_group=depth_per_group,
            num_heads=num_heads,
            window_size=window_size,
            ffn_expansion_ratio=ffn_expansion_ratio,
            aim_reduction_ratio=aim_reduction_ratio,
            group_block_pattern=group_block_pattern,
            drop_path_rate=drop_path_rate,
            upsampler_type=upsampler_type,
            img_range=img_range,
        )

        tags = [
            f"{embed_dim}dim",
        ]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscaling_factor == 1 else "SR",
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=upscaling_factor,
            input_channels=num_in_ch,
            output_channels=num_out_ch,
        )


__all__ = ["LHANArch", "LHAN"]
