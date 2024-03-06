from spandrel_extra_arches.architectures.M3SNet import M3SNet, M3SNetArch

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
        M3SNetArch(),
        lambda: M3SNet(),
        lambda: M3SNet(img_channel=1),
        lambda: M3SNet(width=16),
        lambda: M3SNet(enc_blk_nums=[1, 2, 3, 4], dec_blk_nums=[5, 6, 7, 8]),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/15X6dUwvt1txXA1kOEyEHcvvQ_i7dVlVa/view?usp=drive_link",
        name="m3snet_deblur_model_best_32.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1CJeRsQr-Jzxs_a5pe85OyTZZHxzUj7Ws/view?usp=drive_link",
        name="m3snet_deblur_model_best_64.pth",
    )
    assert_size_requirements(file.load_model())


def test_deblur_model_best_32(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/15X6dUwvt1txXA1kOEyEHcvvQ_i7dVlVa/view?usp=drive_link",
        name="m3snet_deblur_model_best_32.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, M3SNet)
    assert_image_inference(
        file,
        model,
        [TestImage.BLURRY_FACE],
    )


def test_deblur_model_best_64(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1CJeRsQr-Jzxs_a5pe85OyTZZHxzUj7Ws/view?usp=drive_link",
        name="m3snet_deblur_model_best_64.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, M3SNet)
    assert_image_inference(
        file,
        model,
        [TestImage.BLURRY_FACE],
    )
