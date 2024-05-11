from spandrel.architectures.PLKSR import PLKSR, PLKSRArch, RealPLKSR

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
        PLKSRArch(),
        lambda: PLKSR(),
        lambda: PLKSR(dim=32),
        lambda: PLKSR(upscaling_factor=2),
        lambda: PLKSR(n_blocks=6),
        lambda: PLKSR(kernel_size=9),
        lambda: RealPLKSR(),
        lambda: RealPLKSR(dim=32),
        lambda: RealPLKSR(upscaling_factor=2),
        lambda: RealPLKSR(n_blocks=6),
        lambda: RealPLKSR(kernel_size=9),
        lambda: RealPLKSR(use_ea=False),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/12ek1vitEporWc5qqaYo6AMy0-RYlRqu8/view",
        name="4x_realplksr_mssim_pretrain.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1PA3QElJYlgpPYKl0zQ9_D1pnuYfC1vtt/view",
        name="PLKSR_X2_DIV2K.pth",
    )
    assert_size_requirements(file.load_model())


def test_PLKSR_official_x4(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1YtoOjK7vsfVrHFqWGvdjFNcOay8WUuJ7/view",
        name="PLKSR_X4_DIV2K.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, PLKSR)


def test_PLKSR_official_x3(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1W8phbKFTOYL-AnlMJnjx2NWDHZY8jVWW/view",
        name="PLKSR_X3_DIV2K.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, PLKSR)


def test_PLKSR_official_x2(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1PA3QElJYlgpPYKl0zQ9_D1pnuYfC1vtt/view",
        name="PLKSR_X2_DIV2K.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, PLKSR)


def test_RealPLKSR_4x(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/12ek1vitEporWc5qqaYo6AMy0-RYlRqu8/view",
        name="4x_realplksr_mssim_pretrain.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, PLKSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealPLKSR_2x(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1GAdf5VOqYa5ntswT9sYsKKZ2Z7OQp7gO/view",
        name="2x_realplksr_mssim_pretrain.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, PLKSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
