from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.aura_sr import UnetUpsampler as AuraSR


class AuraSRArch(Architecture[AuraSR]):
    def __init__(self) -> None:
        super().__init__(
            id="AuraSR",
            detect=KeyCondition.has_all(
                "mid_attn.layers.0.0.to_out.weight",
                "mid_block2.block1.proj.weights",
                "final_res_block.block1.proj.weights",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[AuraSR]:
        # Defaults
        dim = 64
        image_size = None  # Doesn't need to be specified ahead of time.
        input_image_size = None  # Doesn't need to be specified ahead of time.
        init_dim = None
        out_dim = None  # Doesn't do anything.
        style_network = {
            "dim_in": 128,
            "dim_out": 512,
            "depth": 4,
            # lr_mul doesn't affect the state dict
            # dim_text_latent only affects the state dict in the same way as dim_in
        }
        up_dim_mults = (1, 2, 4, 8, 16)
        down_dim_mults = (4, 8, 16)
        channels = 3
        resnet_block_groups = 8  # Doesn't do anything.
        full_attn = (False, False, False, True, True)
        flash_attn = True  # Not detectable from state_dict.
        self_attn_dim_head = 64
        self_attn_heads = 8  # Not detectable independently from self_attn_dim_head
        attn_depths = (2, 2, 2, 2, 4)
        mid_attn_depth = 4
        num_conv_kernels = 4
        resize_mode = "bilinear"  # Doesn't do anything.
        unconditional = True  # Doesn't do anything.
        skip_connect_scale = 0.4  # Not detectable from state_dict.

        dim = state_dict["final_to_rgb.weight"].shape[1]
        init_dim = state_dict["ups.4.1.0.conv.bias"].shape[0]
        style_network["dim_in"] = state_dict["style_network.net.0.weight"].shape[1]
        style_network["dim_out"] = state_dict["style_network.net.0.weight"].shape[0]
        style_network["depth"] = (get_seq_len(state_dict, "style_network.net") + 1) // 2
        up_dim_mults = (
            state_dict["ups.4.1.0.conv.weight"].shape[1] // dim,
            state_dict["ups.3.1.0.conv.weight"].shape[1] // dim,
            state_dict["ups.2.1.0.conv.weight"].shape[1] // dim,
            state_dict["ups.1.1.0.conv.weight"].shape[1] // dim,
            state_dict["ups.0.1.0.conv.weight"].shape[1] // dim,
        )  # It looks like this tuple should always be of length 5, since making it longer doesn't produce an "ups.5" sequence.
        down_dim_mults = tuple(
            [
                state_dict[f"downs.{i}.1.1.weight"].shape[0] // dim
                for i in range(get_seq_len(state_dict, "downs"))
            ]
        )
        channels = state_dict["final_to_rgb.weight"].shape[0]
        full_attn = (
            "ups.4.1.1.layers.0.0.to_qkv.weight" in state_dict,
            "ups.3.1.1.layers.0.0.to_qkv.weight" in state_dict,
            "ups.2.1.1.layers.0.0.to_qkv.weight" in state_dict,
            "ups.1.1.1.layers.0.0.to_qkv.weight" in state_dict,
            "ups.0.1.1.layers.0.0.to_qkv.weight" in state_dict,
        )
        self_attn_dim_head = (
            state_dict["mid_attn.layers.0.0.to_out.weight"].shape[1] // self_attn_heads
        )
        attn_depths = (
            get_seq_len(state_dict, "ups.4.1.1.layers") if full_attn[0] else 2,
            get_seq_len(state_dict, "ups.3.1.1.layers") if full_attn[1] else 2,
            get_seq_len(state_dict, "ups.2.1.1.layers") if full_attn[2] else 2,
            get_seq_len(state_dict, "ups.1.1.1.layers") if full_attn[3] else 2,
            get_seq_len(state_dict, "ups.0.1.1.layers") if full_attn[4] else 2,
        )
        mid_attn_depth = get_seq_len(state_dict, "mid_attn.layers")
        num_conv_kernels = state_dict["mid_block1.block1.proj.weights"].shape[0]

        model = AuraSR(
            dim=dim,
            image_size=image_size,
            input_image_size=input_image_size,
            init_dim=init_dim,
            out_dim=out_dim,
            style_network=style_network,
            up_dim_mults=up_dim_mults,
            down_dim_mults=down_dim_mults,
            channels=channels,
            resnet_block_groups=resnet_block_groups,
            full_attn=full_attn,
            flash_attn=flash_attn,
            self_attn_dim_head=self_attn_dim_head,
            self_attn_heads=self_attn_heads,
            attn_depths=attn_depths,
            mid_attn_depth=mid_attn_depth,
            num_conv_kernels=num_conv_kernels,
            resize_mode=resize_mode,
            unconditional=unconditional,
            skip_connect_scale=skip_connect_scale,
        )

        scale = style_network["dim_out"] // style_network["dim_in"]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="SR",
            tags=[],
            supports_half=False,  # Probably this is fixable with some mild effort.
            supports_bfloat16=False,
            scale=scale,
            input_channels=channels,
            output_channels=channels,
            size_requirements=SizeRequirements(
                minimum=16,  # I don't know if the minimum can be made even lower, surely less than 16 is pointless?
                multiple_of=8,
            ),
        )


__all__ = ["AuraSRArch", "AuraSR"]
