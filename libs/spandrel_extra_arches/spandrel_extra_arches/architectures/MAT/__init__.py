from typing_extensions import override

from spandrel import (
    Architecture,
    MaskedImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from spandrel.util import KeyCondition

from .__arch.MAT import MAT


class MATArch(Architecture[MAT]):
    def __init__(self) -> None:
        super().__init__(
            id="MAT",
            detect=KeyCondition.has_any(
                "synthesis.first_stage.conv_first.conv.resample_filter",
                "model.synthesis.first_stage.conv_first.conv.resample_filter",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> MaskedImageModelDescriptor[MAT]:
        in_nc = 3
        out_nc = 3

        def map_key(k: str) -> str:
            if k.startswith(("synthesis.", "mapping.")):
                return "model." + k
            return k

        state = {map_key(k): v for k, v in state_dict.items()}

        model = MAT()

        return MaskedImageModelDescriptor(
            model,
            state,
            architecture=self,
            purpose="Inpainting",
            tags=[],
            supports_half=False,
            supports_bfloat16=True,
            input_channels=in_nc,
            output_channels=out_nc,
            size_requirements=SizeRequirements(
                minimum=512, multiple_of=512, square=True
            ),
        )


__all__ = ["MATArch", "MAT"]
