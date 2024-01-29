from spandrel.architectures.DCTLSA import DCTLSA, load

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
        load,
        lambda: DCTLSA(in_nc=3, nf=55, num_modules=6, out_nc=3, upscale=4),
        lambda: DCTLSA(in_nc=3, nf=40, num_modules=6, out_nc=4, upscale=2),
        lambda: DCTLSA(in_nc=4, nf=20, num_modules=7, out_nc=3, upscale=1),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1EQ4rcOleDPSIACSdXKbZSlehyJM5sDKh/view?usp=drive_link",
        name="4x_dctlsa_pretrained.pth",
    )
    assert_size_requirements(file.load_model(), max_candidates=32)


def test_x4(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1EQ4rcOleDPSIACSdXKbZSlehyJM5sDKh/view?usp=drive_link",
        name="4x_dctlsa_pretrained.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DCTLSA)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
