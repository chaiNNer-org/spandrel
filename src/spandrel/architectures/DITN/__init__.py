# ruff: noqa: N806

import math

from ...__helpers.model_descriptor import (
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from ..__arch_helpers.state import get_seq_len
from .arch.DITN_Real import DITN_Real as DITN


def load(state_dict: StateDict) -> ImageModelDescriptor[DITN]:
    # default values
    inp_channels = 3
    dim = 60
    ITL_blocks = 4
    SAL_blocks = 4
    UFONE_blocks = 1
    ffn_expansion_factor = 2
    bias = False
    LayerNorm_type = "WithBias"  # unused internally
    patch_size = 8  # cannot be deduced from state_dict
    upscale = 4

    inp_channels = state_dict["sft.weight"].shape[1]
    dim = state_dict["sft.weight"].shape[0]

    UFONE_blocks = get_seq_len(state_dict, "UFONE")
    ITL_blocks = get_seq_len(state_dict, "UFONE.0.ITLs")
    SAL_blocks = get_seq_len(state_dict, "UFONE.0.SALs")

    ffn_expansion_factor = (
        state_dict["UFONE.0.ITLs.0.ffn.project_in.weight"].shape[0] / 2 / dim
    )

    bias = "UFONE.0.ITLs.0.attn.project_out.bias" in state_dict

    upscale = int(math.sqrt(state_dict["upsample.0.weight"].shape[0] / 3))

    model = DITN(
        inp_channels=inp_channels,
        dim=dim,
        ITL_blocks=ITL_blocks,
        SAL_blocks=SAL_blocks,
        UFONE_blocks=UFONE_blocks,
        ffn_expansion_factor=ffn_expansion_factor,
        bias=bias,
        LayerNorm_type=LayerNorm_type,
        patch_size=patch_size,
        upscale=upscale,
    )

    return ImageModelDescriptor(
        model,
        state_dict,
        architecture="DITN",
        purpose="Restoration" if upscale == 1 else "SR",
        tags=[f"{60}dim"],
        supports_half=True,
        supports_bfloat16=True,
        scale=upscale,
        input_channels=inp_channels,
        output_channels=3,  # hard-coded in the architecture
        size_requirements=SizeRequirements(multiple_of=patch_size),
    )
