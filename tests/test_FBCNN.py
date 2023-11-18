from spandrel import ModelLoader
from spandrel.architectures.FBCNN import FBCNN

from .util import ModelFile, TestImage, assert_image_inference, disallowed_props


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
