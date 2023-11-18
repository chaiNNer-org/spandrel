from ...__helpers.model_descriptor import (
    FaceSRModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .arch.codeformer import CodeFormer


def load(state_dict: StateDict) -> FaceSRModelDescriptor[CodeFormer]:
    dim_embd = 512
    n_head = 8
    n_layers = 9
    codebook_size = 1024
    latent_size = 256
    connect_list = ["32", "64", "128", "256"]
    fix_modules = ["quantize", "generator"]

    # This is just a guess as I only have one model to look at
    position_emb = state_dict["position_emb"]
    dim_embd = position_emb.shape[1]
    latent_size = position_emb.shape[0]

    try:
        n_layers = len(
            set([x.split(".")[1] for x in state_dict.keys() if "ft_layers" in x])
        )
    except:  # noqa: E722
        pass

    codebook_size = state_dict["quantize.embedding.weight"].shape[0]

    # This is also just another guess
    n_head_exp = state_dict["ft_layers.0.self_attn.in_proj_weight"].shape[0] // dim_embd
    n_head = 2**n_head_exp

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

    return FaceSRModelDescriptor(
        model,
        state_dict,
        architecture="CodeFormer",
        tags=[],
        supports_half=False,
        supports_bfloat16=True,
        scale=8,
        input_channels=in_nc,
        output_channels=out_nc,
        size_requirements=SizeRequirements(minimum=16),
    )
