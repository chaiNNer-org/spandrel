from spandrel.architectures.SPAN import SPAN, SPANArch

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
        SPANArch(),
        lambda: SPAN(num_in_ch=3, num_out_ch=3),
        lambda: SPAN(num_in_ch=1, num_out_ch=3),
        lambda: SPAN(num_in_ch=1, num_out_ch=1),
        lambda: SPAN(num_in_ch=4, num_out_ch=4),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, feature_channels=32),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, feature_channels=64),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, upscale=1),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, upscale=2),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, upscale=4),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, upscale=8),
        lambda: SPAN(num_in_ch=3, num_out_ch=3, norm=False),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/span/4x-spanx4_ch48.pth"
    )
    assert_size_requirements(file.load_model())


def test_train():
    assert_training(SPANArch(), SPAN(num_in_ch=3, num_out_ch=3))


def test_SPAN_x4_ch48(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/span/4x-spanx4_ch48.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SPAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
