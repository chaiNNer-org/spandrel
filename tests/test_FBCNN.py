from spandrel import ModelLoader
from spandrel.architectures.FBCNN import FBCNN, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
)


def test_FBCNN_load():
    assert_loads_correctly(
        load,
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
        condition=lambda a, b: (a.nb == b.nb and a.nc == b.nc),
    )


def test_FBCNN_color(snapshot):
    file = ModelFile.from_url(
        "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, FBCNN)
    assert_image_inference(file, model, [TestImage.JPEG_15, TestImage.SR_8])


def test_FBCNN_gray(snapshot):
    file = ModelFile.from_url(
        "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_gray.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, FBCNN)
