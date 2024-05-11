from __future__ import annotations

import math
from typing import Union

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .arch.PLKSR import PLKSR
from .arch.RealPLKSR import RealPLKSR

_PLKSR = Union[PLKSR, RealPLKSR]


class PLKSRArch(Architecture[_PLKSR]):
    def __init__(self) -> None:
        super().__init__(
            id="PLKSR",
            detect=KeyCondition.has_all(
                "feats.0.weight",
                "feats.1.lk.conv.weight",
                "feats.1.refine.weight",
                KeyCondition.has_any(
                    "feats.1.channe_mixer.0.weight",
                    "feats.1.channel_mixer.0.weight",
                ),
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[_PLKSR]:
        dim = 64
        n_blocks = 28
        scale = 4
        kernel_size = 17
        split_ratio = 0.25
        use_ea = True

        # RealPLKSR only
        norm_groups = 4  # un-detectable
        dropout = 0  # un-detectable

        dim = state_dict["feats.0.weight"].shape[0]

        total_feat_layers = get_seq_len(state_dict, "feats")
        scale = math.isqrt(
            state_dict[f"feats.{total_feat_layers - 1}.weight"].shape[0] // 3
        )

        kernel_size = state_dict["feats.1.lk.conv.weight"].shape[2]
        split_ratio = state_dict["feats.1.lk.conv.weight"].shape[0] / dim

        use_ea = "feats.1.attn.f.0.weight" in state_dict

        # Yes, the normal version has this typo.
        if "feats.1.channe_mixer.0.weight" in state_dict:
            n_blocks = total_feat_layers - 2

            mixer_0_shape = state_dict["feats.1.channe_mixer.0.weight"].shape[2]
            mixer_2_shape = state_dict["feats.1.channe_mixer.2.weight"].shape[2]
            if mixer_0_shape == 3 and mixer_2_shape == 1:
                ccm_type = "CCM"
            elif mixer_0_shape == 3 and mixer_2_shape == 3:
                ccm_type = "DCCM"
            elif mixer_0_shape == 1 and mixer_2_shape == 3:
                ccm_type = "ICCM"
            else:
                raise ValueError("Unknown CCM type")
            more_tags = [ccm_type]

            model = PLKSR(
                dim=dim,
                upscaling_factor=scale,
                n_blocks=n_blocks,
                kernel_size=kernel_size,
                split_ratio=split_ratio,
                use_ea=use_ea,
                ccm_type=ccm_type,
            )
        # and RealPLKSR doesn't. This makes it really convenient to detect.
        elif "feats.1.channel_mixer.0.weight" in state_dict:
            more_tags = ["Real"]

            n_blocks = total_feat_layers - 3
            model = RealPLKSR(
                dim=dim,
                upscaling_factor=scale,
                n_blocks=n_blocks,
                kernel_size=kernel_size,
                split_ratio=split_ratio,
                use_ea=use_ea,
                norm_groups=norm_groups,
                dropout=dropout,
            )
        else:
            raise ValueError("Unknown model type")

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{dim}dim", f"{n_blocks}nb", f"{kernel_size}ks", *more_tags],
            supports_half=False,
            supports_bfloat16=True,
            scale=scale,
            input_channels=3,
            output_channels=3,
        )
