from spandrel.architectures.HAT import HAT, HATArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        HATArch(),
        lambda: HAT(),
        lambda: HAT(in_chans=1),
        lambda: HAT(in_chans=4),
        lambda: HAT(embed_dim=64),
        lambda: HAT(depths=(1, 2, 3, 2, 1), num_heads=(5, 4, 3, 5, 6)),
        lambda: HAT(window_size=13),
        lambda: HAT(compress_ratio=2),
        lambda: HAT(squeeze_factor=15),
        lambda: HAT(overlap_ratio=0.75),
        lambda: HAT(mlp_ratio=3),
        lambda: HAT(qkv_bias=False),
        lambda: HAT(ape=True),
        lambda: HAT(patch_norm=False),
        lambda: HAT(resi_connection="1conv"),
        lambda: HAT(resi_connection="identity"),
        lambda: HAT(upsampler="pixelshuffle", upscale=1),
        lambda: HAT(upsampler="pixelshuffle", upscale=2),
        lambda: HAT(upsampler="pixelshuffle", upscale=3),
        lambda: HAT(upsampler="pixelshuffle", upscale=4),
        lambda: HAT(upsampler="pixelshuffle", upscale=8),
        lambda: HAT(upsampler="pixelshuffle", num_feat=32),
        # there are multiple equivalent squeeze_factor values for each models,
        # so we cannot test for equal value
        ignore_parameters={"squeeze_factor"},
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1Y7-3IgWfIAui9BMQIsFT9CzZXVMlx__e/view?usp=drive_link",
        name="HAT-S_SRx2.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1pdhaO1fJq3tgSqDIbymdDiGxu4S0nqVq/view?usp=drive_link",
        name="HAT_SRx4.pth",
    )
    assert_size_requirements(file.load_model())


def test_HAT_S_2x(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1Y7-3IgWfIAui9BMQIsFT9CzZXVMlx__e/view?usp=drive_link",
        name="HAT-S_SRx2.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_S_3x(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1yBoJkvvvQ5GcPxV8cF0z6KGgOF5pMclX/view?usp=drive_link",
        name="HAT-S_SRx3.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_S_4x(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1YvU9PF1XqlP8TVzH7P0bg-YlfP1TKDPC/view?usp=drive_link",
        name="HAT-S_SRx4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_3x(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1dWG4X_6VUSi1hhIwX0zEwddWI9M0tFmI/view?usp=drive_link",
        name="HAT_SRx3.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_4x(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1pdhaO1fJq3tgSqDIbymdDiGxu4S0nqVq/view?usp=drive_link",
        name="HAT_SRx4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_L_4x(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1uefIctjoNE3Tg6GTzelesTTshVogQdUf/view?usp=drive_link",
        name="HAT-L_SRx4_ImageNet-pretrain.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_community1(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Phhofm/models/raw/main/4xLexicaHAT/4xLexicaHAT.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_community2(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Phhofm/models/raw/main/4xNomos8kSCHAT-S/4xNomos8kSCHAT-S.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
