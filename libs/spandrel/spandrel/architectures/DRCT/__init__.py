import math

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.drct_arch import DRCT


def _get_upscale_pixelshuffle(
    state_dict: StateDict, key_prefix: str = "upsample"
) -> int:
    upscale = 1

    for i in range(0, 10, 2):
        key = f"{key_prefix}.{i}.weight"
        if key not in state_dict:
            break

        shape = state_dict[key].shape
        num_feat = shape[1]
        upscale *= math.isqrt(shape[0] // num_feat)

    return upscale


class DRCTArch(Architecture[DRCT]):
    def __init__(self) -> None:
        super().__init__(
            id="DRCT",
            detect=KeyCondition.has_all(
                "conv_first.weight",
                "conv_first.bias",
                "layers.0.swin1.norm1.weight",
                "layers.0.swin1.norm1.bias",
                "layers.0.swin1.attn.relative_position_bias_table",
                "layers.0.swin1.attn.relative_position_index",
                "layers.0.swin1.attn.qkv.weight",
                "layers.0.swin1.attn.proj.weight",
                "layers.0.swin1.attn.proj.bias",
                "layers.0.swin1.norm2.weight",
                "layers.0.swin1.mlp.fc1.weight",
                "layers.0.swin1.mlp.fc1.bias",
                "layers.0.swin1.mlp.fc2.weight",
                "layers.0.adjust1.weight",
                "layers.0.swin2.norm1.weight",
                "layers.0.adjust2.weight",
                "layers.0.swin3.norm1.weight",
                "layers.0.adjust3.weight",
                "layers.0.swin4.norm1.weight",
                "layers.0.adjust4.weight",
                "layers.0.swin5.norm1.weight",
                "layers.0.adjust5.weight",
                "norm.weight",
                "norm.bias",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[DRCT]:
        # Defaults
        img_size = 64
        patch_size = 1  # cannot be detected
        in_chans = 3
        embed_dim = 180
        depths = (6, 6, 6, 6, 6, 6)
        num_heads = (6, 6, 6, 6, 6, 6)
        window_size = 16
        mlp_ratio = 2.0
        qkv_bias = True
        ape = False
        patch_norm = True
        upscale = 2
        img_range = 1.0  # cannot be deduced from state_dict
        upsampler = ""
        resi_connection = "1conv"
        gc = 32

        # detect
        in_chans = state_dict["conv_first.weight"].shape[1]
        embed_dim = state_dict["conv_first.weight"].shape[0]

        num_layers = get_seq_len(state_dict, "layers")
        depths = (6,) * num_layers
        num_heads = []
        for i in range(num_layers):
            num_heads.append(
                state_dict[f"layers.{i}.swin1.attn.relative_position_bias_table"].shape[
                    1
                ]
            )

        mlp_ratio = state_dict["layers.0.swin1.mlp.fc1.weight"].shape[0] / embed_dim

        window_square = state_dict[
            "layers.0.swin1.attn.relative_position_bias_table"
        ].shape[0]
        window_size = (math.isqrt(window_square) + 1) // 2

        if "conv_last.weight" in state_dict:
            upsampler = "pixelshuffle"
            upscale = _get_upscale_pixelshuffle(state_dict, "upsample")
        else:
            upsampler = ""
            upscale = 1

        if "conv_after_body.weight" in state_dict:
            resi_connection = "1conv"
        else:
            resi_connection = "identity"

        qkv_bias = "layers.0.swin1.attn.qkv.bias" in state_dict
        gc = state_dict["layers.0.adjust1.weight"].shape[0]

        patch_norm = "patch_embed.norm.weight" in state_dict
        ape = "absolute_pos_embed" in state_dict

        if "layers.0.swin2.attn_mask" in state_dict:
            img_size = (
                math.isqrt(state_dict["layers.0.swin2.attn_mask"].shape[0])
                * window_size
                * patch_size
            )
        else:
            # we only know that the input size is <= window_size,
            # so we just assume that the input size is window_size
            img_size = window_size * patch_size

        model = DRCT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            ape=ape,
            patch_norm=patch_norm,
            upscale=upscale,
            img_range=img_range,
            upsampler=upsampler,
            resi_connection=resi_connection,
            gc=gc,
        )

        size_tag = ["large"] if len(depths) >= 10 else []
        tags = [
            *size_tag,
            f"s{img_size}w{window_size}",
            f"{embed_dim}dim",
            f"{resi_connection}",
        ]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscale == 1 else "SR",
            tags=tags,
            supports_half=False,  # Too much weirdness to support this at the moment
            supports_bfloat16=True,
            scale=upscale,
            input_channels=in_chans,
            output_channels=in_chans,
            size_requirements=SizeRequirements(),
        )
