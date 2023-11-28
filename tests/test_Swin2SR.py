from spandrel.architectures.Swin2SR import Swin2SR, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
)


def test_Swin2SR_load():
    assert_loads_correctly(
        load,
        lambda: Swin2SR(window_size=8, upsampler="pixelshuffledirect"),
        lambda: Swin2SR(window_size=8, upsampler="pixelshuffledirect", ape=True),
        lambda: Swin2SR(
            window_size=8,
            upsampler="pixelshuffledirect",
            depths=[6, 7, 5, 3, 4],
            num_heads=[5, 2, 9, 1, 2],
        ),
        lambda: Swin2SR(window_size=8, upsampler="pixelshuffledirect", qkv_bias=False),
        lambda: Swin2SR(
            window_size=8, upsampler="pixelshuffledirect", patch_norm=False
        ),
        lambda: Swin2SR(
            window_size=8, upsampler="pixelshuffledirect", resi_connection="1conv"
        ),
        lambda: Swin2SR(
            window_size=8, upsampler="pixelshuffledirect", resi_connection="3conv"
        ),
        lambda: Swin2SR(window_size=8, upsampler="pixelshuffledirect", patch_size=2),
        condition=lambda a, b: (
            a.img_range == b.img_range
            and a.upscale == b.upscale
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


def test_Swin2SR_2x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Swin2SR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_Swin2SR_4x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X4_64.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Swin2SR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_Swin2SR_compressed(snapshot):
    file = ModelFile.from_url(
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_CompressedSR_X4_48.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Swin2SR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_Swin2SR_jpeg(snapshot):
    file = ModelFile.from_url(
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_Jpeg_dynamic.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Swin2SR)
    # This is a grayscale model for some reason...
    # assert_image_inference(
    #     file,
    #     model,
    #     [TestImage.SR_64, TestImage.JPEG_15],
    # )


def test_Swin2SR_lightweight_2x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_Lightweight_X2_64.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Swin2SR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
