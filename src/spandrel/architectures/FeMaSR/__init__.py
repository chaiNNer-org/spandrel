from __future__ import annotations

from ...__helpers.model_descriptor import (
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from ..__arch_helpers.state import get_first_seq_index, get_seq_len
from .arch.femasr import FeMaSRNet as FeMaSR

_inv_channel_query_dict = {
    256: [8, 16, 32, 64],
    128: [128],
    64: [256],
    32: [512],
}


def _clean_state_dict(state_dict: StateDict):
    # To make my day a little brighter, the pretrained models of FeMaSR have a bunch
    # of useless keys in their state dict. With great delight, I saw those keys cause
    # errors when calling `model.load_state_dict(state_dict)`, so this function
    # removes them.

    keys = list(state_dict.keys())
    for k in keys:
        if k.startswith(("sft_fusion_group.", "multiscale_encoder.upsampler.")):
            del state_dict[k]


def load(state_dict: StateDict) -> ImageModelDescriptor[FeMaSR]:
    _clean_state_dict(state_dict)

    # in_channel = 3
    # codebook_params: list[list[int]] = [[32, 1024, 512]]
    # gt_resolution = 256
    # LQ_stage = False
    # norm_type = "gn"
    act_type = "silu"
    use_quantize = True  # cannot be deduced from state_dict
    # scale_factor = 4
    # use_semantic_loss = False
    use_residual = True  # cannot be deduced from state_dict

    in_channel = state_dict["multiscale_encoder.in_conv.weight"].shape[1]
    use_semantic_loss = "conv_semantic.0.weight" in state_dict

    # gt_resolution can be derived from the decoders
    # we assume that gt_resolution is a power of 2
    max_depth = get_seq_len(state_dict, "decoder_group")
    # in the last decoder iteration, we essentially have:
    #   out_ch = channel_query_dict[gt_resolution]
    out_ch = state_dict[f"decoder_group.{max_depth-1}.block.1.weight"].shape[0]
    gt_resolution_candidates = _inv_channel_query_dict[out_ch]
    # just choose the largest one
    gt_resolution = gt_resolution_candidates[-1]

    # the codebook is complex to reconstruct
    cb_height = get_seq_len(state_dict, "quantize_group")
    codebook_params = []
    for i in range(cb_height):
        emb_num = state_dict[f"quantize_group.{i}.embedding.weight"].shape[0]
        emb_dim = state_dict[f"quantize_group.{i}.embedding.weight"].shape[1]

        # scale_in_ch = channel_query_dict[self.codebook_scale[scale]]
        scale_in_ch = state_dict[f"after_quant_group.{i}.conv.weight"].shape[0]
        candidates = _inv_channel_query_dict[scale_in_ch]
        # we just need *a* scale, so we can pick the first one
        codebook_params.append([candidates[0], emb_num, emb_dim])

    #   max_depth = int(log2(gt_resolution // scale_0))
    # We assume that gt_resolution and scale_0 are powers of 2, so we can calculate
    # them directly
    scale_0 = gt_resolution // (2**max_depth)
    codebook_params[0][0] = scale_0

    # scale factor
    swin_block_index = get_first_seq_index(
        state_dict,
        "multiscale_encoder.blocks.{}.swin_blks.0.residual_group.blocks.0.attn.relative_position_bias_table",
    )
    if swin_block_index >= 0:
        LQ_stage = True  # noqa: N806
        # encode_depth = int(log2(gt_resolution // scale_factor // scale_0))
        encode_depth = swin_block_index
        scale_factor = gt_resolution // (2**encode_depth * scale_0)
    else:
        LQ_stage = False  # noqa: N806
        scale_factor = 1

    if "decoder_group.0.block.2.conv.0.norm.running_mean" in state_dict:
        norm_type = "bn"
    elif "decoder_group.0.block.2.conv.0.norm.weight" in state_dict:
        norm_type = "gn"
    else:
        # we cannot differentiate between "none" and "in"
        norm_type = "in"

    model = FeMaSR(
        in_channel=in_channel,
        codebook_params=codebook_params,
        gt_resolution=gt_resolution,
        LQ_stage=LQ_stage,
        norm_type=norm_type,
        act_type=act_type,
        use_quantize=use_quantize,
        scale_factor=scale_factor,
        use_semantic_loss=use_semantic_loss,
        use_residual=use_residual,
    )

    multiple_of = {2: 32, 4: 16}.get(scale_factor, 1)

    return ImageModelDescriptor(
        model,
        state_dict,
        architecture="FeMaSR",
        purpose="Restoration" if scale_factor == 1 else "SR",
        tags=[],
        supports_half=True,  # TODO
        supports_bfloat16=True,  # TODO
        scale=scale_factor,
        input_channels=in_channel,
        output_channels=in_channel,
        size_requirements=SizeRequirements(multiple_of=multiple_of),
        call_fn=lambda model, image: model(image)[0],
    )
