from ...__helpers.model_descriptor import (
    MaskedImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.MAT import MAT


def load(state_dict: StateDict) -> MaskedImageModelDescriptor[MAT]:
    in_nc = 3
    out_nc = 3

    state = {
        k.replace("synthesis", "model.synthesis").replace("mapping", "model.mapping"): v
        for k, v in state_dict.items()
    }

    model = MAT()

    return MaskedImageModelDescriptor(
        model,
        state,
        architecture="MAT",
        purpose="Inpaint",
        tags=[],
        supports_half=False,
        supports_bfloat16=True,
        input_channels=in_nc,
        output_channels=out_nc,
        size_requirements=SizeRequirements(minimum=512, multiple_of=512, square=True),
    )
