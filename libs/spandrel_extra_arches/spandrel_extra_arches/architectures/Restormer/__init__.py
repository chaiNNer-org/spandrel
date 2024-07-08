from __future__ import annotations

from typing_extensions import override

from spandrel import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from spandrel.util import KeyCondition, get_seq_len

from .__arch.restormer_arch import Restormer


class RestormerArch(Architecture[Restormer]):
    def __init__(self) -> None:
        super().__init__(
            id="Restormer",
            detect=KeyCondition.has_all(
                "patch_embed.proj.weight",
                "encoder_level1.0.norm1.body.weight",
                "encoder_level1.0.attn.temperature",
                "encoder_level1.0.attn.qkv.weight",
                "encoder_level1.0.attn.qkv_dwconv.weight",
                "encoder_level1.0.attn.project_out.weight",
                "encoder_level1.0.norm2.body.weight",
                "encoder_level1.0.ffn.project_in.weight",
                "encoder_level1.0.ffn.dwconv.weight",
                "encoder_level1.0.ffn.project_out.weight",
                "down1_2.body.0.weight",
                "encoder_level2.0.attn.temperature",
                "down2_3.body.0.weight",
                "encoder_level3.0.attn.temperature",
                "down3_4.body.0.weight",
                "latent.0.attn.temperature",
                "up4_3.body.0.weight",
                "reduce_chan_level3.weight",
                "decoder_level3.0.attn.temperature",
                "up3_2.body.0.weight",
                "reduce_chan_level2.weight",
                "decoder_level2.0.attn.temperature",
                "up2_1.body.0.weight",
                "decoder_level1.0.attn.temperature",
                "refinement.0.norm1.body.weight",
                "refinement.0.attn.temperature",
                "output.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[Restormer]:
        inp_channels = 3
        out_channels = 3
        dim = 48
        num_blocks = [4, 6, 6, 8]
        num_refinement_blocks = 4
        heads = [1, 2, 4, 8]
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = "WithBias"  # noqa: N806
        dual_pixel_task = False

        inp_channels = state_dict["patch_embed.proj.weight"].shape[1]
        out_channels = state_dict["output.weight"].shape[0]
        dim = state_dict["patch_embed.proj.weight"].shape[0]

        num_blocks[0] = get_seq_len(state_dict, "encoder_level1")
        num_blocks[1] = get_seq_len(state_dict, "encoder_level2")
        num_blocks[2] = get_seq_len(state_dict, "encoder_level3")
        num_blocks[3] = get_seq_len(state_dict, "latent")

        num_refinement_blocks = get_seq_len(state_dict, "refinement")

        heads[0] = state_dict["encoder_level1.0.attn.temperature"].shape[0]
        heads[1] = state_dict["encoder_level2.0.attn.temperature"].shape[0]
        heads[2] = state_dict["encoder_level3.0.attn.temperature"].shape[0]
        heads[3] = state_dict["latent.0.attn.temperature"].shape[0]

        # hidden_dim = int(dim * ffn_expansion_factor)
        hidden_dim = state_dict["encoder_level1.0.ffn.project_out.weight"].shape[1]
        if hidden_dim == int(dim * 2.66):
            # this is needed to get the exact value
            ffn_expansion_factor = 2.66
        else:
            ffn_expansion_factor = hidden_dim / dim

        bias = "encoder_level1.0.attn.qkv.bias" in state_dict
        dual_pixel_task = "skip_conv.weight" in state_dict

        LayerNorm_type = (  # noqa: N806
            "WithBias"
            if "encoder_level1.0.norm1.body.bias" in state_dict
            else "BiasFree"
        )

        model = Restormer(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[f"{dim}dim"],
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=1,
            input_channels=inp_channels,
            output_channels=out_channels,
            size_requirements=SizeRequirements(multiple_of=8),
        )


__all__ = ["RestormerArch", "Restormer"]
