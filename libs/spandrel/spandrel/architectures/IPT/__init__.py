from __future__ import annotations

from typing import Sequence

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.ipt import IPT


class IPTArch(Architecture[IPT]):
    def __init__(self) -> None:
        super().__init__(
            id="IPT",
            detect=KeyCondition.has_all(
                "sub_mean.weight",
                "sub_mean.bias",
                "add_mean.weight",
                "add_mean.bias",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
                "weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[IPT]:
        patch_size: int
        patch_dim: int
        # n_feats: int
        rgb_range: float = 1
        scale: Sequence[int] = [1]
        num_heads: int
        # num_layers: int
        num_queries: int
        dropout_rate: float = 0
        mlp = True
        pos_every = False
        no_pos = False
        no_norm = False

        n_feats = state_dict["head.0.0.weight"].shape[0]

        num_layers = get_seq_len(state_dict, "body.encoder.layers")

        model = IPT(
            patch_size=patch_size,
            patch_dim=patch_dim,
            n_feats=n_feats,
            rgb_range=rgb_range,
            n_colors=3,  # must be RGB
            scale=scale,
            num_heads=num_heads,
            num_layers=num_layers,
            num_queries=num_queries,
            dropout_rate=dropout_rate,
            mlp=mlp,
            pos_every=pos_every,
            no_pos=no_pos,
            no_norm=no_norm,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[],
            supports_half=False,
            supports_bfloat16=True,
            scale=1,
            input_channels=3,  # only supports RGB
            output_channels=3,
            size_requirements=SizeRequirements(),
        )
