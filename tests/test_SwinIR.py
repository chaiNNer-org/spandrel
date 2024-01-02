from spandrel.architectures.SwinIR import SwinIR, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
)


def test_SwinIR_load():
    assert_loads_correctly(
        load,
        lambda: SwinIR(window_size=8),
        lambda: SwinIR(depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], window_size=8),
        lambda: SwinIR(depths=[6, 6, 2, 1], num_heads=[6, 4, 6, 3], window_size=8),
        condition=lambda a, b: (
            a.img_range == b.img_range
            and (not a.upsampler or a.upscale == b.upscale)
            and a.upsampler == b.upsampler
            and a.window_size == b.window_size
            and a.num_layers == b.num_layers
            and a.embed_dim == b.embed_dim
            and a.ape == b.ape
            and a.patch_norm == b.patch_norm
            and a.num_features == b.num_features
            and a.mlp_ratio == b.mlp_ratio
            and a.patches_resolution == b.patches_resolution
        ),
    )


def test_SwinIR_M_s64w8_2x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwinIR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_SwinIR_M_s48w8_4x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwinIR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_SwinIR_S_2x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwinIR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_SwinIR_L_4x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwinIR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
