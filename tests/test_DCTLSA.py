from spandrel.architectures.DCTLSA import DCTLSA, DCTLSAArch

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
        DCTLSAArch(),
        lambda: DCTLSA(in_nc=3, nf=55, num_modules=6, out_nc=3, upscale=4),
        lambda: DCTLSA(in_nc=3, nf=40, num_modules=6, out_nc=4, upscale=2),
        lambda: DCTLSA(in_nc=4, nf=20, num_modules=7, out_nc=3, upscale=1),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/zengkun301/DCTLSA/raw/main/pretrained/X4.pt",
        name="4x_dctlsa.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/zengkun301/DCTLSA/raw/main/pretrained/X2.pt",
        name="2x_dctlsa.pth",
    )
    assert_size_requirements(file.load_model())


def test_x4(snapshot):
    file = ModelFile.from_url(
        "https://github.com/zengkun301/DCTLSA/raw/main/pretrained/X4.pt",
        name="4x_dctlsa.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DCTLSA)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_x2(snapshot):
    file = ModelFile.from_url(
        "https://github.com/zengkun301/DCTLSA/raw/main/pretrained/X2.pt",
        name="2x_dctlsa.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DCTLSA)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
