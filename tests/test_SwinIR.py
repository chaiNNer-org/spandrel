from spandrel import ModelLoader
from spandrel.architectures.SwinIR import SwinIR

from .util import ImageTestNames, ModelFile, compare_images_to_results, disallowed_props


def test_SwinIR_M_s64w8_2x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwinIR)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )


def test_SwinIR_M_s48w8_4x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwinIR)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )


def test_SwinIR_S_2x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwinIR)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )


def test_SwinIR_L_4x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwinIR)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )
