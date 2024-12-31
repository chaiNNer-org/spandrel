from spandrel.architectures.Sebica import Sebica, SebicaArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        SebicaArch(),
        lambda: Sebica(num_in_ch=1, num_out_ch=3),
        lambda: Sebica(num_in_ch=1, num_out_ch=1),
        lambda: Sebica(num_in_ch=3, num_out_ch=3),
        lambda: Sebica(num_in_ch=4, num_out_ch=4),
        lambda: Sebica(num_in_ch=3, num_out_ch=3, num_feat=8),
        lambda: Sebica(num_in_ch=3, num_out_ch=3, num_feat=16),
        lambda: Sebica(num_in_ch=3, num_out_ch=3, sr_rate=1),
        lambda: Sebica(num_in_ch=3, num_out_ch=3, sr_rate=2),
        lambda: Sebica(num_in_ch=3, num_out_ch=3, sr_rate=4),
        lambda: Sebica(num_in_ch=3, num_out_ch=3, sr_rate=8),
        lambda: Sebica(num_in_ch=3, num_out_ch=3),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1hqR2niCRzgUnOmdGLcFJL3s7wP_uYKdD/view?usp=sharing",
        name="2x_sebica_eva.pth",
    )
    assert_size_requirements(file.load_model())


def test_2x_sebica_eva(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1hqR2niCRzgUnOmdGLcFJL3s7wP_uYKdD/view?usp=sharing",
        name="2x_sebica_eva.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Sebica)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
