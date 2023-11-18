from ...__helpers.model_descriptor import (
    InpaintModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.LaMa import LaMa


def load(state_dict: StateDict) -> InpaintModelDescriptor[LaMa]:
    in_nc = 4
    out_nc = 3

    state = {
        k.replace("generator.model", "model.model"): v for k, v in state_dict.items()
    }

    model = LaMa(
        in_nc=in_nc,
        out_nc=out_nc,
    )

    return InpaintModelDescriptor(
        model,
        state,
        architecture="LaMa",
        tags=[],
        supports_half=False,
        supports_bfloat16=True,
        input_channels=in_nc,
        output_channels=out_nc,
        size_requirements=SizeRequirements(minimum=16),
    )
