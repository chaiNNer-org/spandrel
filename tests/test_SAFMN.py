from spandrel.architectures.SAFMN import SAFMN, SAFMNArch

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
        SAFMNArch(),
        lambda: SAFMN(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=4),
        lambda: SAFMN(dim=36, n_blocks=8, ffn_scale=3.0, upscaling_factor=3),
        lambda: SAFMN(dim=8, n_blocks=3, ffn_scale=5.0, upscaling_factor=2),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1TCcCJBlI88vG18292LL_iG5pBAR84Zaw/view?usp=drive_link",
        name="SAFMN_DF2K_x2.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1lEMemKXUvDYLOU69gvlJQZKB_x0GccFh/view?usp=drive_link",
        name="SAFMN_DF2K_x3.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1OG1ihKt0Ka_h7VKZVjmOCBVmG_68w-hP/view?usp=drive_link",
        name="SAFMN_DF2K_x4.pth",
    )
    assert_size_requirements(file.load_model())


def test_train():
    assert_training(SAFMNArch(), SAFMN(dim=8))


def test_SAFMN_DF2K_x2(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1TCcCJBlI88vG18292LL_iG5pBAR84Zaw/view?usp=drive_link",
        name="SAFMN_DF2K_x2.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SAFMN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_SAFMN_DF2K_x3(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1lEMemKXUvDYLOU69gvlJQZKB_x0GccFh/view?usp=drive_link",
        name="SAFMN_DF2K_x3.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SAFMN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_SAFMN_DF2K_x4(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1OG1ihKt0Ka_h7VKZVjmOCBVmG_68w-hP/view?usp=drive_link",
        name="SAFMN_DF2K_x4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SAFMN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
