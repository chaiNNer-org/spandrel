from ...__helpers.model_descriptor import (
    MaskedImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from ..__arch_helpers.state import get_seq_len
from .arch.LaMa import LaMa


def load(state_dict: StateDict) -> MaskedImageModelDescriptor[LaMa]:
    state_dict = {
        k.replace("generator.model", "model.model"): v for k, v in state_dict.items()
    }

    in_nc = 4
    out_nc = 3

    in_nc = state_dict["model.model.1.ffc.convl2l.weight"].shape[1]

    seq_len = get_seq_len(state_dict, "model.model")
    out_nc = state_dict[f"model.model.{seq_len - 1}.weight"].shape[0]

    model = LaMa(
        in_nc=in_nc,
        out_nc=out_nc,
    )

    return MaskedImageModelDescriptor(
        model,
        state_dict,
        architecture="LaMa",
        purpose="Inpaint",
        tags=[],
        supports_half=False,
        supports_bfloat16=True,
        input_channels=in_nc,
        output_channels=out_nc,
        size_requirements=SizeRequirements(minimum=16),
    )
