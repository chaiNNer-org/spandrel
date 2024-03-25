from typing_extensions import override

from spandrel import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from spandrel.util import KeyCondition, get_seq_len

from .arch.codeformer import CodeFormer


class CodeFormerArch(Architecture[CodeFormer]):
    def __init__(self) -> None:
        super().__init__(
            id="CodeFormer",
            detect=KeyCondition.has_all(
                "position_emb",
                "idx_pred_layer.1.weight",
                "quantize.embedding.weight",
                "ft_layers.0.self_attn.in_proj_weight",
                "encoder.blocks.0.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[CodeFormer]:
        dim_embd = 512
        n_head = 8  # cannot be deduced from state dict
        n_layers = 9
        codebook_size = 1024
        latent_size = 256
        connect_list = ["32", "64", "128", "256"]
        fix_modules = ["quantize", "generator"]

        dim_embd = state_dict["position_emb"].shape[1]
        latent_size = state_dict["position_emb"].shape[0]
        codebook_size = state_dict["idx_pred_layer.1.weight"].shape[0]
        n_layers = get_seq_len(state_dict, "ft_layers")

        keys = ["16", "32", "64", "128", "256", "512"]
        connect_list = list(
            filter(lambda k: f"fuse_convs_dict.{k}.scale.0.weight" in state_dict, keys)
        )

        in_nc = state_dict["encoder.blocks.0.weight"].shape[1]

        in_nc = in_nc
        out_nc = in_nc

        model = CodeFormer(
            dim_embd=dim_embd,
            n_head=n_head,
            n_layers=n_layers,
            codebook_size=codebook_size,
            latent_size=latent_size,
            connect_list=connect_list,
            fix_modules=fix_modules,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="FaceSR",
            tags=[],
            supports_half=False,
            supports_bfloat16=True,
            scale=1,
            input_channels=in_nc,
            output_channels=out_nc,
            size_requirements=SizeRequirements(multiple_of=512, square=True),
            call_fn=lambda model, image: model(image)[0],
        )
