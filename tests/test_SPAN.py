from spandrel.architectures.SPAN import SPAN, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
)


def test_SPAN_load():
    assert_loads_correctly(
        load,
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
        condition=lambda a, b: (
            a.in_channels == b.in_channels
            and a.out_channels == b.out_channels
            and a.img_range == b.img_range
        ),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-spanx4-ch48.pth"
    )
    assert_size_requirements(file.load_model())


def test_SPAN_x4_ch48(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-spanx4-ch48.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SPAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
