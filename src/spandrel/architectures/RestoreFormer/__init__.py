from ...__helpers.model_descriptor import (
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from ..__arch_helpers.state import get_seq_len
from .arch.restoreformer_arch import RestoreFormer


def load(state_dict: StateDict) -> ImageModelDescriptor[RestoreFormer]:
    n_embed = 1024
    embed_dim = 256
    ch = 64
    out_ch = 3
    ch_mult = (1, 2, 2, 4, 4, 8)
    num_res_blocks = 2
    attn_resolutions = (16,)
    dropout = 0.0
    in_channels = 3
    resolution = 512
    z_channels = 256
    double_z = False
    enable_mid = True
    head_size = 8  # cannot be deduced from the shape of tensors in state_dict

    n_embed = state_dict["quantize.embedding.weight"].shape[0]
    embed_dim = state_dict["quantize.embedding.weight"].shape[1]
    z_channels = state_dict["quant_conv.weight"].shape[1]
    double_z = state_dict["encoder.conv_out.weight"].shape[0] == 2 * z_channels

    enable_mid = "encoder.mid.block_1.norm1.weight" in state_dict

    ch = state_dict["encoder.conv_in.weight"].shape[0]
    in_channels = state_dict["encoder.conv_in.weight"].shape[1]
    out_ch = state_dict["decoder.conv_out.weight"].shape[0]

    num_res_blocks = get_seq_len(state_dict, "encoder.down.0.block")

    ch_mult_len = get_seq_len(state_dict, "encoder.down")
    ch_mult_list = [1] * ch_mult_len
    for i in range(ch_mult_len):
        ch_mult_list[i] = (
            state_dict[f"encoder.down.{i}.block.0.conv2.weight"].shape[0] // ch
        )
    ch_mult = tuple(ch_mult_list)

    model = RestoreFormer(
        n_embed=n_embed,
        embed_dim=embed_dim,
        ch=ch,
        out_ch=out_ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout,
        in_channels=in_channels,
        resolution=resolution,
        z_channels=z_channels,
        double_z=double_z,
        enable_mid=enable_mid,
        head_size=head_size,
    )

    return ImageModelDescriptor(
        model,
        state_dict,
        architecture="RestoreFormer",
        purpose="FaceSR",
        tags=[],
        supports_half=False,
        supports_bfloat16=True,
        scale=8,
        input_channels=in_channels,
        output_channels=out_ch,
        size_requirements=SizeRequirements(minimum=16),
    )
