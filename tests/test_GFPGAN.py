from spandrel.architectures.GFPGAN import GFPGAN, GFPGANArch
from tests.test_CodeFormer import assert_loads_correctly

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_training,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        GFPGANArch(),
        lambda: GFPGAN(),
    )


def test_train():
    # TODO: fix training
    assert_training(GFPGANArch(), GFPGAN())


def test_GFPGAN_1_2(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GFPGAN)


def test_GFPGAN_1_3(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GFPGAN)


def test_GFPGAN_1_4(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GFPGAN)
    assert_image_inference(
        model_file=file,
        model=model,
        test_images=[TestImage.BLURRY_FACE],
    )
