from spandrel.architectures.SCUNet import SCUNet, load

from .util import (
    ModelFile,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
)


def test_SCUNet_load():
    assert_loads_correctly(
        load,
        lambda: SCUNet(),
        lambda: SCUNet(in_nc=1),
        lambda: SCUNet(in_nc=4),
        lambda: SCUNet(dim=32),
        lambda: SCUNet(dim=24),
        lambda: SCUNet(config=[5, 3, 7, 2, 3, 1, 3]),
        condition=lambda a, b: a.dim == b.dim and a.config == b.config,
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth"
    )
    assert_size_requirements(file.load_model(), max_size=128)

    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
    )
    assert_size_requirements(file.load_model(), max_size=128)


def test_SCUNet_color_GAN(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SCUNet)


def test_SCUNet_color_RSNR(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SCUNet)


def test_SCUNet_color_25(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_25.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SCUNet)


def test_SCUNet_gray_25(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_gray_25.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SCUNet)
