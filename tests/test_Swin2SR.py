from spandrel.architectures.Swin2SR import Swin2SR

from .util import ModelFile, TestImage, assert_image_inference, disallowed_props


def test_Swin2SR_4x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X4_64.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Swin2SR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
