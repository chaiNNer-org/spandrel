from spandrel.architectures.GFPGAN import GFPGANArch, GFPGANv1Clean
from tests.test_CodeFormer import assert_loads_correctly

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    disallowed_props,
)


def test_load():
    assert_loads_correctly(
        GFPGANArch(),
        lambda: GFPGANv1Clean(),
    )


def test_GFPGAN_1_2(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GFPGANv1Clean)


def test_GFPGAN_1_3(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GFPGANv1Clean)


def test_GFPGAN_1_4(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GFPGANv1Clean)
    assert_image_inference(
        model_file=file,
        model=model,
        test_images=[TestImage.BLURRY_FACE],
    )
