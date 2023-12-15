from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from ...__helpers.model_descriptor import (
    ImageModelDescriptor,
    ModelTiling,
    StateDict,
)
from ..__arch_helpers.color import lab_to_rgb, linear_rgb_to_lab, rgb_to_linear_rgb
from ..__arch_helpers.state import get_seq_len
from .arch.ddcolor import DDColor


def _call(model: DDColor, input: Tensor) -> Tensor:
    if model.num_output_channels != 2:
        raise ValueError("The call API for DDColor only supports 2 output channels")

    # input tensor to have the shape (1, 1, H, W)
    if len(input.shape) != 4 or input.shape[0] != 1 or input.shape[1] != 1:
        raise ValueError(
            f"The call API for DDColor expected the input tensor to be of the shape (1, 1, H, W), but found {input.shape}"
        )

    h = input.shape[2]
    w = input.shape[3]

    # This implements the pipeline from the original DDColor code adopted to tensor format:
    # https://github.com/piddnad/DDColor/blob/e13093cd8a5c9667b32d358fb0c3473661e6031c/inference/colorization_pipline.py#L58-L82

    input_gray_linear = rgb_to_linear_rgb(input)
    # (1, 3, H, W)
    input_rgb_linear = torch.hstack([input_gray_linear] * 3)

    # (1, 1, H, W)
    orig_l = linear_rgb_to_lab(input_rgb_linear)[:, :1, :, :]

    # (1, 3, M, M) (M = model.input_size)
    img = F.interpolate(input_rgb_linear, size=model.input_size, mode="bilinear")
    # (1, 1, M, M)
    img_l = linear_rgb_to_lab(img)[:, :1, :, :]
    img_zero = torch.zeros_like(img_l)
    # (1, 3, M, M)
    img_gray_lab = torch.hstack([img_l, img_zero, img_zero])
    img_gray_rgb = lab_to_rgb(img_gray_lab)

    # (1, 2, M, M)
    output_ab = model(img_gray_rgb)

    # resize ab -> concat original l -> rgb
    # (1, 2, H, W)
    output_ab_resize = F.interpolate(output_ab, size=(h, w), mode="bilinear")
    output_lab = torch.hstack([orig_l, output_ab_resize])
    output_rgb = lab_to_rgb(output_lab)

    # (1, 3, H, W)
    return output_rgb


def _get_encoder_convnext_depths_and_dims(state_dict: StateDict):
    depths = [
        get_seq_len(state_dict, "encoder.arch.stages.0"),
        get_seq_len(state_dict, "encoder.arch.stages.1"),
        get_seq_len(state_dict, "encoder.arch.stages.2"),
        get_seq_len(state_dict, "encoder.arch.stages.3"),
    ]
    dims = [
        state_dict["encoder.arch.stages.0.0.norm.weight"].shape[0],
        state_dict["encoder.arch.stages.1.0.norm.weight"].shape[0],
        state_dict["encoder.arch.stages.2.0.norm.weight"].shape[0],
        state_dict["encoder.arch.stages.3.0.norm.weight"].shape[0],
    ]
    return depths, dims


def load(state_dict: StateDict) -> ImageModelDescriptor[DDColor]:
    # they commented out head_cls, so we have to clean it up
    state_dict.pop("encoder.arch.head_cls.weight", None)
    state_dict.pop("encoder.arch.head_cls.bias", None)

    # default values
    encoder_name = "convnext-l"
    decoder_name = "MultiScaleColorDecoder"
    num_input_channels = 3  # has to be 3 or else DDColor will error
    input_size = (256, 256)
    nf = 512
    num_output_channels = 3
    last_norm: Literal["Batch", "BatchZero", "Weight", "Spectral"] = "Weight"
    do_normalize = False  # cannot be deduced from state_dict
    num_queries = 256
    num_scales = 3
    dec_layers = 9

    num_output_channels = state_dict["refine_net.0.0.weight_orig"].shape[0]
    num_queries = state_dict["refine_net.0.0.weight_orig"].shape[1] - 3

    if "decoder.last_shuf.conv.1.weight" in state_dict:
        last_norm = "Batch"  # or "BatchZero", but we can't tell which
        nf = state_dict["decoder.last_shuf.conv.1.weight"].shape[0] // 8
    elif "decoder.last_shuf.conv.0.weight_orig" in state_dict:
        last_norm = "Spectral"
        nf = state_dict["decoder.last_shuf.conv.0.weight_orig"].shape[0] // 8
    elif "decoder.last_shuf.conv.0.weight_g" in state_dict:
        last_norm = "Weight"
        nf = state_dict["decoder.last_shuf.conv.0.weight_g"].shape[0] // 8
    else:
        raise ValueError("Unknown last norm")

    if "decoder.color_decoder.query_feat.weight" in state_dict:
        decoder_name = "MultiScaleColorDecoder"
        num_scales = state_dict["decoder.color_decoder.level_embed.weight"].shape[0]
        dec_layers = get_seq_len(
            state_dict, "decoder.color_decoder.transformer_self_attention_layers"
        )
    else:
        decoder_name = "SingleColorDecoder"
        # num_scales and dec_layers aren't defined for SingleColorDecoder,
        # so we just assume default values
        num_scales = 3
        dec_layers = 9

    depths, dims = _get_encoder_convnext_depths_and_dims(state_dict)
    if depths == [3, 3, 9, 3] and dims == [96, 192, 384, 768]:
        encoder_name = "convnext-t"
    elif depths == [3, 3, 27, 3] and dims == [96, 192, 384, 768]:
        encoder_name = "convnext-s"
    elif depths == [3, 3, 27, 3] and dims == [128, 256, 512, 1024]:
        encoder_name = "convnext-b"
    elif depths == [3, 3, 27, 3] and dims == [192, 384, 768, 1536]:
        encoder_name = "convnext-l"
    else:
        raise ValueError("Unknown encoder architecture")

    model = DDColor(
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        num_input_channels=num_input_channels,
        input_size=input_size,
        nf=nf,
        num_output_channels=num_output_channels,
        last_norm=last_norm,
        do_normalize=do_normalize,
        num_queries=num_queries,
        num_scales=num_scales,
        dec_layers=dec_layers,
    )

    return ImageModelDescriptor(
        model,
        state_dict,
        architecture="DDColor",
        purpose="Restoration",
        tags=[
            decoder_name,
            last_norm,
            encoder_name,
            f"{nf}nf",
            f"{num_queries}nq {num_scales}ns {dec_layers}dl",
        ],
        supports_half=False,
        supports_bfloat16=True,
        scale=1,
        input_channels=1,
        output_channels=3,
        tiling=ModelTiling.INTERNAL,
        call_fn=_call,
    )
