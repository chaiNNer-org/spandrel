from spandrel.architectures.DnCNN import DnCNN, DnCNNArch

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
        DnCNNArch(),
        lambda: DnCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode="BR"),
        lambda: DnCNN(in_nc=3, out_nc=3, nc=32, nb=15, act_mode="R"),
        lambda: DnCNN(in_nc=4, out_nc=3, nc=32, nb=15, act_mode="R", mode="FDnCNN"),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth",
    )
    assert_size_requirements(file.load_model())


def test_train():
    assert_training(DnCNNArch(), DnCNN())


def test_dncnn_color_blind(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DnCNN)
    assert_image_inference(
        file,
        model,
        [TestImage.JPEG_15],
    )


def test_dncnn_50(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_50.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DnCNN)


def test_fdncnn_color(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/fdncnn_color.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DnCNN)
    assert_image_inference(
        file,
        model,
        [TestImage.JPEG_15],
    )
