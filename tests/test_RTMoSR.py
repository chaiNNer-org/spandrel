from spandrel.architectures.RTMoSR import RTMoSR, RTMoSRArch

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
        RTMoSRArch(),
        lambda: RTMoSR(scale=2),
        lambda: RTMoSR(scale=4),
        lambda: RTMoSR(scale=8),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/184r0aTa68l4dOgPhaHkd0mdLVw7J-DmL/view?usp=drive_link",
        name="2x_umzi_mahou_rtmosr.pth",
    )
    assert_size_requirements(file.load_model())


def test_2x_umzi_mahou_rtmosr(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/184r0aTa68l4dOgPhaHkd0mdLVw7J-DmL/view?usp=drive_link",
        name="2x_umzi_mahou_rtmosr.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RTMoSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
