from __future__ import annotations

import math
from typing import Sequence

from typing_extensions import override

from spandrel import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from spandrel.util import KeyCondition, get_seq_len

from .arch.vrt import VRT


def _get_pa_frames(state_dict: StateDict) -> int:
    if "stage1.pa_deform.conv_offset.0.weight" not in state_dict:
        return 0
    dim0 = state_dict["stage1.pa_deform.conv_offset.0.weight"].shape[0]
    conv_in = state_dict["stage1.pa_deform.conv_offset.0.weight"].shape[1]
    return 2 * (conv_in - dim0) // (dim0 + 2)


def _get_depth_mul(depths: Sequence[int], muls: Sequence[int]) -> float:
    def test(mul: float) -> bool:
        return all(m == int(mul * d) for d, m in zip(depths, muls))

    candidates = [0.75, 0.25, 0.5, 1.0, *(i / 10 for i in range(1, 10))]
    for candidate in candidates:
        if test(candidate):
            return candidate

    raise ValueError("Could not find a suitable depth multiplier")


def _get_window_size(state_dict: StateDict) -> list[int]:
    # window_size = d, h, w
    # a = 3 * (2h - 1) * (2w - 1)
    a: int = state_dict[
        "stage1.residual_group1.blocks.0.attn.relative_position_bias_table"
    ].shape[0]
    # b = 2wh
    b: int = state_dict[
        "stage1.residual_group1.blocks.0.attn.relative_position_index"
    ].shape[0]
    # c = dwh
    c: int = state_dict[
        "stage1.residual_group2.blocks.0.attn.relative_position_index"
    ].shape[0]

    def is_square(n: int) -> bool:
        return math.isqrt(n) ** 2 == n

    if not is_square(a // 3) or not is_square(b // 2):
        raise RuntimeError(
            "Unsupported window size. Only window sizes with width == height are supported."
        )

    # we can now assume h == w
    h = math.isqrt(b // 2)
    w = h
    d = c // (w * h)

    return [d, h, w]


class VRTArch(Architecture[VRT]):
    def __init__(self) -> None:
        super().__init__(
            id="VRT",
            detect=KeyCondition.has_all(
                "conv_first.weight",
                "conv_first.bias",
                "norm.weight",
                "conv_after_body.weight",
                KeyCondition.has_any(
                    # with pa_frames
                    KeyCondition.has_all(
                        "spynet.mean",
                        "spynet.std",
                        "spynet.basic_module.0.basic_module.0.weight",
                        "spynet.basic_module.0.basic_module.0.bias",
                        "spynet.basic_module.0.basic_module.8.bias",
                        "spynet.basic_module.5.basic_module.8.bias",
                        "conv_last.weight",
                    ),
                    # without pa_frames
                    KeyCondition.has_all(
                        "linear_fuse.weight",
                        "conv_last.weight",
                    ),
                ),
                "stage1.reshape.1.weight",
                "stage1.linear1.weight",
                "stage1.linear2.weight",
                "stage1.residual_group1.blocks.0.norm1.weight",
                "stage1.residual_group1.blocks.0.attn.relative_position_bias_table",
                "stage1.residual_group1.blocks.0.attn.relative_position_index",
                "stage1.residual_group1.blocks.0.attn.qkv_self.weight",
                "stage1.residual_group1.blocks.0.attn.proj.weight",
                "stage1.residual_group1.blocks.0.mlp.fc11.weight",
                "stage1.residual_group1.blocks.0.mlp.fc12.weight",
                "stage1.residual_group1.blocks.0.mlp.fc2.weight",
                "stage2.reshape.2.weight",
                "stage7.reshape.2.weight",
                "stage8.0.1.weight",
                "stage8.0.2.weight",
                "stage8.1.linear.weight",
                "stage8.1.residual_group.blocks.0.norm1.weight",
                "stage8.1.residual_group.blocks.0.attn.relative_position_bias_table",
                "stage8.1.residual_group.blocks.0.attn.relative_position_index",
                "stage8.1.residual_group.blocks.0.attn.qkv_self.weight",
                "stage8.1.residual_group.blocks.0.attn.proj.weight",
                "stage8.1.residual_group.blocks.0.mlp.fc11.weight",
                "stage8.1.residual_group.blocks.0.mlp.fc12.weight",
                "stage8.1.residual_group.blocks.0.mlp.fc2.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[VRT]:
        # Default
        upscale = 1
        in_chans = 3
        out_chans = 3
        img_size = [6, 64, 64]  # mostly unused
        window_size: Sequence[int] = [6, 8, 8]
        depths: list[int] = [8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4]
        indep_reconsts: list[int] = [11, 12]
        embed_dims: list[int] = [
            120,
            120,
            120,
            120,
            120,
            120,
            120,
            180,
            180,
            180,
            180,
            180,
            180,
        ]
        num_heads: list[int] = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        mul_attn_ratio = 0.75
        mlp_ratio = 2.0
        qkv_bias = True
        qk_scale = None  # cannot be deduced from state dict
        drop_path_rate = 0.2  # cannot be deduced from state dict
        pa_frames = 2
        deformable_groups = 16
        nonblind_denoising = False

        # detect
        pa_frames = _get_pa_frames(state_dict)

        # in_chans & nonblind_denoising
        nonblind_denoising = False
        conv_first_in_chans = state_dict["conv_first.weight"].shape[1]
        if pa_frames:
            if conv_first_in_chans % 9 == 0:
                in_chans = conv_first_in_chans // 9
            else:
                in_chans = (conv_first_in_chans - 1) // 9
                nonblind_denoising = True
        else:
            in_chans = conv_first_in_chans

        # upscale
        upscale = 1
        if pa_frames and "conv_before_upsample.0.weight" in state_dict:
            for i in range(10):
                key = f"upsample.{i*5}.weight"
                if key in state_dict:
                    shape = state_dict[key].shape
                    upscale *= math.isqrt(shape[0] // shape[1])
                else:
                    break

        out_chans = state_dict["conv_last.weight"].shape[0]

        deformable_groups = 16
        if pa_frames:
            deformable_groups = (
                state_dict["stage1.pa_deform.conv_offset.6.weight"].shape[0] // 27
            )

        qkv_bias = "stage1.residual_group1.blocks.0.attn.qkv_self.bias" in state_dict

        depth_len = get_seq_len(state_dict, "stage8") - 1 + 7
        depths: list[int] = []
        depths_muls: list[int] = []
        embed_dims: list[int] = []
        num_heads: list[int] = []
        for i in range(7):
            depth_mul = get_seq_len(state_dict, f"stage{i+1}.residual_group1.blocks")
            depths_muls.append(depth_mul)
            depth = depth_mul + get_seq_len(
                state_dict, f"stage{i+1}.residual_group2.blocks"
            )
            depths.append(depth)
            embed_dims.append(
                state_dict[f"stage{i+1}.linear1.weight"].shape[0],
            )
            num_heads.append(
                state_dict[
                    f"stage{i+1}.residual_group1.blocks.0.attn.relative_position_bias_table"
                ].shape[1],
            )
        for i in range(7, depth_len):
            depths.append(
                get_seq_len(state_dict, f"stage8.{i+1-7}.residual_group.blocks"),
            )
            embed_dims.append(
                state_dict[f"stage8.{i+1-7}.linear.weight"].shape[0],
            )
            num_heads.append(
                state_dict[
                    f"stage8.{i+1-7}.residual_group.blocks.1.attn.relative_position_bias_table"
                ].shape[1],
            )

        mul_attn_ratio = _get_depth_mul(depths, depths_muls)

        window_size = _get_window_size(state_dict)

        indep_reconsts: list[int] = []
        for i in range(7, depth_len):
            wh = state_dict[
                f"stage8.{i+1-7}.residual_group.blocks.0.attn.relative_position_index"
            ].shape[0]
            if wh == window_size[1] * window_size[2]:
                indep_reconsts.append(i)

        mlp_ratio = (
            state_dict["stage1.residual_group1.blocks.0.mlp.fc11.weight"].shape[0]
            / embed_dims[0]
        )

        if not pa_frames:
            img_size[0] = state_dict["linear_fuse.weight"].shape[1] // embed_dims[0]

        model = VRT(
            upscale=upscale,
            in_chans=in_chans,
            out_chans=out_chans,
            img_size=img_size,
            window_size=window_size,
            depths=depths,
            indep_reconsts=indep_reconsts,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mul_attn_ratio=mul_attn_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_path_rate=drop_path_rate,
            pa_frames=pa_frames,
            deformable_groups=deformable_groups,
            nonblind_denoising=nonblind_denoising,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscale == 1 else "SR",
            tags=[],
            supports_half=False,  # Too much weirdness to support this at the moment
            supports_bfloat16=True,
            scale=upscale,
            input_channels=in_chans,
            output_channels=out_chans,
            size_requirements=SizeRequirements(),
        )
