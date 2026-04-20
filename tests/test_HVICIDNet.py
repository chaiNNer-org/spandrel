from spandrel.architectures.HVICIDNet import HVICIDNet, HVICIDNetArch

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
        HVICIDNetArch(),
        lambda: HVICIDNet(),
        lambda: HVICIDNet(channels=[12, 24, 36, 48]),
        lambda: HVICIDNet(heads=[1, 4, 16, 1]),
        lambda: HVICIDNet(norm=True),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1KbRpPL9-TfBDmZoV6S05pTvMB0sm_OwO/view?usp=sharing",
        name="HVI-CIDNet_lolv1_w_perc.pth",
    )
    assert_size_requirements(file.load_model())


def test_train():
    assert_training(HVICIDNetArch(), HVICIDNet())


def test_LOLv1(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1KbRpPL9-TfBDmZoV6S05pTvMB0sm_OwO/view?usp=sharing",
        name="HVI-CIDNet_lolv1_w_perc.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HVICIDNet)
    assert_image_inference(
        file,
        model,
        [TestImage.LOW_LIGHT_FIVE_K],
    )
