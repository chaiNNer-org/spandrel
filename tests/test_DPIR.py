from spandrel.architectures.DPIR import IRCNN, DnCNN, DPIRArch

from .util import (
    ModelFile,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
)


def test_load():
    assert_loads_correctly(
        DPIRArch(),
        lambda: DnCNN(in_nc=1, out_nc=3, nc=64, nb=17, act_mode="BR"),
        lambda: DnCNN(in_nc=3, out_nc=4, nc=32, nb=15, act_mode="R"),
        lambda: IRCNN(in_nc=1, out_nc=1, nc=64),
        lambda: IRCNN(in_nc=3, out_nc=1, nc=32),
        lambda: IRCNN(in_nc=1, out_nc=4, nc=32),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/ircnn_color.pth",
    )
    assert_size_requirements(file.load_model())


def test_dncnn_color_blind(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DnCNN)


def test_ircnn_color(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/ircnn_color.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, IRCNN)


def test_ircnn_gray(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/ircnn_gray.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, IRCNN)