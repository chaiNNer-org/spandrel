from spandrel.architectures.SwinIR import SwinIR

from .util import ModelFile, TestImage, assert_image_inference, disallowed_props


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
