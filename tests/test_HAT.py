from spandrel import ModelLoader
from spandrel.architectures.HAT import HAT

from .util import ModelFile, TestImage, assert_image_inference, disallowed_props


def test_HAT_community1(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Phhofm/models/raw/main/4xLexicaHAT/4xLexicaHAT.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


# TODO: We don't support HAT-S models yet

# def test_HAT_community2(snapshot):
#     file = ModelFile.from_url(
#         "https://github.com/Phhofm/models/raw/main/4xNomos8kSCHAT-S/4xNomos8kSCHAT-S.pth"
#     )
#     model = ModelLoader().load_from_file(file.path)
#     assert model == snapshot(exclude=disallowed_props)
#     assert isinstance(model.model, HAT)
