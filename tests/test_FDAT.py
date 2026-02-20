from spandrel.architectures.FDAT import FDAT, FDATArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
)


def test_load():
    assert_loads_correctly(
        FDATArch(),
        lambda: FDAT(upsampler_type="pixelshuffle"),
        lambda: FDAT(upsampler_type="transpose+conv"),
        lambda: FDAT(upsampler_type="nearest+conv"),
        lambda: FDAT(upsampler_type="dysample"),
        lambda: FDAT(upsampler_type="pixelshuffledirect"),
        lambda: FDAT(upsampler_type="pixelshuffle", scale=1),
        lambda: FDAT(upsampler_type="pixelshuffle", scale=2),
        lambda: FDAT(upsampler_type="pixelshuffle", scale=3),
        lambda: FDAT(upsampler_type="pixelshuffle", scale=8),
        lambda: FDAT(upsampler_type="pixelshuffle", scale=1, num_in_ch=1, num_out_ch=1),
        lambda: FDAT(upsampler_type="transpose+conv", scale=1),
        lambda: FDAT(upsampler_type="transpose+conv", scale=2),
        lambda: FDAT(upsampler_type="transpose+conv", scale=3),
        lambda: FDAT(
            upsampler_type="transpose+conv",
            scale=2,
            num_in_ch=1,
            num_out_ch=1,
        ),
        lambda: FDAT(upsampler_type="nearest+conv", scale=1),
        lambda: FDAT(upsampler_type="nearest+conv", scale=2),
        lambda: FDAT(upsampler_type="nearest+conv", scale=3),
        lambda: FDAT(upsampler_type="nearest+conv", scale=8),
        lambda: FDAT(num_groups=6),
        lambda: FDAT(depth_per_group=6),
        lambda: FDAT(num_heads=6),
        lambda: FDAT(ffn_expansion_ratio=1.5),
        lambda: FDAT(aim_reduction_ratio=7),
        lambda: FDAT(mid_dim=48),
        lambda: FDAT(
            scale=2,
            embed_dim=96,
            num_groups=2,
            depth_per_group=2,
            num_heads=3,
            window_size=8,
            ffn_expansion_ratio=1.5,
            aim_reduction_ratio=8,
            group_block_pattern=None,
            upsampler_type="pixelshuffle",
            img_range=1.0,
        ),
        lambda: FDAT(scale=2, unshuffle_mod=True),
        lambda: FDAT(scale=1, unshuffle_mod=True),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/the-database/traiNNer-redux/releases/download/pretrained-models/2x_DF2K_FDAT_M_500k_fp16.safetensors",
        name="2x_DF2K_FDAT_M_500k_fp16.safetensors",
    )
    assert_size_requirements(file.load_model())


def test_fdat_m(snapshot):
    file = ModelFile.from_url(
        "https://github.com/the-database/traiNNer-redux/releases/download/pretrained-models/2x_DF2K_FDAT_M_500k_fp16.safetensors",
        name="2x_DF2K_FDAT_M_500k_fp16.safetensors",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, FDAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
