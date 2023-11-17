from ...__helpers.model_descriptor import (
    FaceSRModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.restoreformer_arch import RestoreFormer


def load(state_dict: StateDict) -> FaceSRModelDescriptor[RestoreFormer]:
    in_nc = 3
    out_nc = 3

    model = RestoreFormer(
        in_channels=in_nc,
        out_ch=out_nc,
    )

    return FaceSRModelDescriptor(
        model,
        state_dict,
        architecture="RestoreFormer",
        tags=[],
        supports_half=False,
        supports_bfloat16=True,
        scale=8,
        input_channels=in_nc,
        output_channels=out_nc,
        size=SizeRequirements(minimum=16),
    )
