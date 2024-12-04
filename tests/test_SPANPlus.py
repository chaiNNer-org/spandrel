from spandrel.architectures.SPANPlus import SPANPlus, SPANPlusArch

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
        SPANPlusArch(),
        lambda: SPANPlus(num_in_ch=3, num_out_ch=3),
        lambda: SPANPlus(num_in_ch=1, num_out_ch=3),
        lambda: SPANPlus(num_in_ch=1, num_out_ch=1),
        lambda: SPANPlus(num_in_ch=4, num_out_ch=4),
        lambda: SPANPlus(num_in_ch=3, num_out_ch=3, feature_channels=32),
        lambda: SPANPlus(num_in_ch=3, num_out_ch=3, feature_channels=64),
        lambda: SPANPlus(num_in_ch=3, num_out_ch=3, upscale=1),
        lambda: SPANPlus(num_in_ch=3, num_out_ch=3, upscale=2),
        lambda: SPANPlus(num_in_ch=3, num_out_ch=3, upscale=4),
        lambda: SPANPlus(num_in_ch=3, num_out_ch=3, upscale=8),
        lambda: SPANPlus(num_in_ch=3, num_out_ch=3),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/TNTwise/SPAN-ncnn-vulkan/releases/download/20240626-224116/2x_spanplus.pth"
    )
    assert_size_requirements(file.load_model())


def test_SPANPlus_x2_ch48(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TNTwise/SPAN-ncnn-vulkan/releases/download/20240626-224116/2x_spanplus.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SPANPlus)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
