from spandrel.architectures.DRUNet import DRUNet, DRUNetArch

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
        DRUNetArch(),
        lambda: DRUNet(),
        lambda: DRUNet(in_nc=4, out_nc=3),
        lambda: DRUNet(nc=[32, 128, 512, 32]),
        lambda: DRUNet(nb=2),
        lambda: DRUNet(downsample_mode="strideconv"),
        lambda: DRUNet(downsample_mode="avgpool"),
        lambda: DRUNet(upsample_mode="upconv"),
        lambda: DRUNet(upsample_mode="pixelshuffle"),
        lambda: DRUNet(upsample_mode="convtranspose"),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth",
    )
    assert_size_requirements(file.load_model())


def test_train():
    assert_training(DRUNetArch(), DRUNet())


def test_drunet_color(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DRUNet)
    assert_image_inference(
        file,
        model,
        [TestImage.JPEG_15],
    )
