from spandrel_extra_arches.architectures.DDColor import DDColor, DDColorArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        DDColorArch(),
        lambda: DDColor(
            nf=64,
            num_queries=20,
            input_size=(32, 32),
            encoder_name="convnext-l",
            decoder_name="MultiScaleColorDecoder",
            num_output_channels=3,
            last_norm="Weight",
            num_scales=3,
            dec_layers=9,
        ),
        lambda: DDColor(
            nf=64,
            num_queries=20,
            input_size=(32, 32),
            encoder_name="convnext-b",
            decoder_name="MultiScaleColorDecoder",
            num_output_channels=2,
            last_norm="Spectral",
            num_scales=3,
            dec_layers=9,
        ),
        lambda: DDColor(
            nf=64,
            num_queries=20,
            input_size=(32, 32),
            encoder_name="convnext-s",
            decoder_name="MultiScaleColorDecoder",
            num_output_channels=2,
            last_norm="Batch",
            num_scales=3,
            dec_layers=9,
        ),
        lambda: DDColor(
            nf=64,
            num_queries=20,
            input_size=(32, 32),
            encoder_name="convnext-t",
            decoder_name="SingleColorDecoder",
            num_output_channels=2,
            last_norm="Spectral",
            num_scales=3,
            dec_layers=9,
        ),
        ignore_parameters={"input_size"},
    )


def test_DDColor_paper_tiny(snapshot):
    file = ModelFile.from_url(
        "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_paper_tiny.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DDColor)
    assert_image_inference(
        file,
        model,
        [TestImage.GRAY_EINSTEIN],
    )
