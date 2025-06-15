from spandrel.architectures.LHAN import LHAN, LHANArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
)


def test_load():
    assert_loads_correctly(
        LHANArch(),
        lambda: LHAN(upsampler_type="pixelshuffle"),
        lambda: LHAN(upsampler_type="transpose_conv"),
        lambda: LHAN(upsampler_type="nearest_conv"),
        lambda: LHAN(upsampler_type="pixelshuffle", upscaling_factor=1),
        lambda: LHAN(upsampler_type="pixelshuffle", upscaling_factor=2),
        lambda: LHAN(upsampler_type="pixelshuffle", upscaling_factor=3),
        lambda: LHAN(upsampler_type="pixelshuffle", upscaling_factor=8),
        lambda: LHAN(
            upsampler_type="pixelshuffle", upscaling_factor=1, num_in_ch=1, num_out_ch=1
        ),
        lambda: LHAN(upsampler_type="transpose_conv", upscaling_factor=2),
        lambda: LHAN(upsampler_type="transpose_conv", upscaling_factor=3),
        lambda: LHAN(
            upsampler_type="transpose_conv",
            upscaling_factor=2,
            num_in_ch=1,
            num_out_ch=1,
        ),
        # lambda: LHAN(upsampler_type="nearest_conv", upscaling_factor=1),
        # lambda: LHAN(upsampler_type="nearest_conv", upscaling_factor=2),
        # lambda: LHAN(upsampler_type="nearest_conv", upscaling_factor=3),
        # lambda: LHAN(upsampler_type="nearest_conv", upscaling_factor=8),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1CZaPYhO1EQRodctgPWKeC2yKbI6nX2-B/view?usp=sharing",
        name="RCAN_BIX4.safetensors",
    )
    assert_size_requirements(file.load_model())


def test_rcan_bix4(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1CZaPYhO1EQRodctgPWKeC2yKbI6nX2-B/view?usp=sharing",
        name="RCAN_BIX4.safetensors",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, LHAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
