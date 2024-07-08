from __future__ import annotations

from typing_extensions import override

from spandrel import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from spandrel.util import KeyCondition, get_seq_len

from .__arch.MPRNet import MPRNet


class MPRNetArch(Architecture[MPRNet]):
    def __init__(self) -> None:
        super().__init__(
            id="MPRNet",
            detect=KeyCondition.has_all(
                "shallow_feat1.0.weight",
                "shallow_feat1.1.CA.conv_du.0.weight",
                "shallow_feat1.1.CA.conv_du.2.weight",
                "shallow_feat1.1.body.0.weight",
                "shallow_feat1.1.body.2.weight",
                "shallow_feat2.0.weight",
                "shallow_feat3.0.weight",
                "stage1_encoder.encoder_level1.0.CA.conv_du.0.weight",
                "stage1_encoder.encoder_level1.0.CA.conv_du.2.weight",
                "stage1_encoder.encoder_level1.0.body.2.weight",
                "stage1_encoder.encoder_level1.1.body.2.weight",
                "stage1_encoder.encoder_level2.1.body.2.weight",
                "stage1_encoder.encoder_level3.0.CA.conv_du.0.weight",
                "stage1_decoder.decoder_level1.0.CA.conv_du.0.weight",
                "stage1_decoder.decoder_level1.0.body.0.weight",
                "stage1_decoder.decoder_level2.0.CA.conv_du.0.weight",
                "stage1_decoder.decoder_level3.0.CA.conv_du.0.weight",
                "stage1_decoder.skip_attn1.CA.conv_du.0.weight",
                "stage1_decoder.skip_attn2.CA.conv_du.0.weight",
                "stage1_decoder.up32.up.1.weight",
                "stage2_encoder.encoder_level1.0.CA.conv_du.0.weight",
                "stage2_decoder.decoder_level1.0.CA.conv_du.0.weight",
                "sam12.conv1.weight",
                "sam12.conv3.weight",
                "sam23.conv3.weight",
                "concat12.weight",
                "concat23.weight",
                "tail.weight",
                "stage3_orsnet.orb1.body.0.CA.conv_du.0.weight",
                "stage3_orsnet.orb1.body.0.CA.conv_du.2.weight",
                "stage3_orsnet.orb1.body.0.body.0.weight",
                "stage3_orsnet.orb1.body.0.body.2.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[MPRNet]:
        # in_c: int = 3
        # out_c: int = 3
        # n_feat: int = 40
        # scale_unetfeats: int = 20
        # scale_orsnetfeats: int = 16
        # num_cab: int = 8
        # kernel_size: int = 3
        # reduction = 4
        # bias = False

        in_c = state_dict["shallow_feat1.0.weight"].shape[1]
        n_feat = state_dict["shallow_feat1.0.weight"].shape[0]
        kernel_size = state_dict["shallow_feat1.0.weight"].shape[2]
        bias = "shallow_feat1.0.bias" in state_dict
        reduction = n_feat // state_dict["shallow_feat1.1.CA.conv_du.0.weight"].shape[0]

        out_c = state_dict["tail.weight"].shape[0]
        scale_orsnetfeats = state_dict["tail.weight"].shape[1] - n_feat
        scale_unetfeats = (
            state_dict["stage1_encoder.encoder_level2.0.CA.conv_du.0.weight"].shape[1]
            - n_feat
        )

        num_cab = get_seq_len(state_dict, "stage3_orsnet.orb1.body") - 1

        model = MPRNet(
            in_c=in_c,
            out_c=out_c,
            n_feat=n_feat,
            scale_unetfeats=scale_unetfeats,
            scale_orsnetfeats=scale_orsnetfeats,
            num_cab=num_cab,
            kernel_size=kernel_size,
            reduction=reduction,
            bias=bias,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[f"{n_feat}nf"],
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=1,
            input_channels=in_c,
            output_channels=out_c,
            size_requirements=SizeRequirements(multiple_of=8),
            call_fn=lambda model, x: model(x)[0],
        )


__all__ = ["MPRNetArch", "MPRNet"]
