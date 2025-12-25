from spandrel.architectures.SwinIR import SwinIR, SwinIRArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    assert_training,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        SwinIRArch(),
        lambda: SwinIR(window_size=8),
        lambda: SwinIR(depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], window_size=8),
        lambda: SwinIR(depths=[6, 6, 2, 1], num_heads=[6, 4, 6, 3], window_size=8),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
    )
    assert_size_requirements(file.load_model())


def test_train():
    # TODO: fix training
    assert_training(SwinIRArch(), SwinIR())


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


# Model is from https://github.com/XPixelGroup/DiffBIR
def test_SwinIR_DiffBIR_General(snapshot):
    file = ModelFile.from_url(
        "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwinIR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
