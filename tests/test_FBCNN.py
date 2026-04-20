from spandrel.architectures.FBCNN import FBCNN, FBCNNArch

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
        FBCNNArch(),
        lambda: FBCNN(),
        lambda: FBCNN(in_nc=1),
        lambda: FBCNN(out_nc=4),
        lambda: FBCNN(nb=3),
        lambda: FBCNN(nc=[16, 32, 64, 16]),
        lambda: FBCNN(downsample_mode="strideconv"),
        lambda: FBCNN(downsample_mode="avgpool"),
        lambda: FBCNN(upsample_mode="convtranspose"),
        lambda: FBCNN(upsample_mode="upconv"),
        lambda: FBCNN(upsample_mode="pixelshuffle"),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth"
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_gray.pth"
    )
    assert_size_requirements(file.load_model())


def test_train():
    assert_training(FBCNNArch(), FBCNN())


def test_FBCNN_color(snapshot):
    file = ModelFile.from_url(
        "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, FBCNN)
    assert_image_inference(file, model, [TestImage.JPEG_15, TestImage.SR_8])


def test_FBCNN_gray(snapshot):
    file = ModelFile.from_url(
        "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_gray.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, FBCNN)
