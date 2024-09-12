from spandrel.architectures.USRNet import USRNet, USRNetArch

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
        USRNetArch(),
        lambda: USRNet(),
        lambda: USRNet(in_nc=2, out_nc=1),
        lambda: USRNet(nb=5),
        lambda: USRNet(nc=[16, 32, 32, 8]),
        lambda: USRNet(h_nc=16),
        lambda: USRNet(n_iter=4),
        lambda: USRNet(downsample_mode="maxpool"),
        lambda: USRNet(downsample_mode="strideconv"),
        lambda: USRNet(upsample_mode="upconv"),
        lambda: USRNet(upsample_mode="pixelshuffle"),
        lambda: USRNet(upsample_mode="convtranspose"),
    )


def test_size_requirements():
    return
    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
    )
    assert_size_requirements(file.load_model())


def test_SwinIR_M_s64w8_2x(snapshot):
    return
    file = ModelFile.from_url(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, USRNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
