from spandrel.architectures.SwiftSRGAN import SwiftSRGAN, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
)


def test_SwiftSRGAN_load():
    assert_loads_correctly(
        load,
        lambda: SwiftSRGAN(),
        lambda: SwiftSRGAN(in_channels=1),
        lambda: SwiftSRGAN(num_channels=32),
        lambda: SwiftSRGAN(num_blocks=7),
        lambda: SwiftSRGAN(upscale_factor=2),
        lambda: SwiftSRGAN(upscale_factor=4),
        lambda: SwiftSRGAN(upscale_factor=8),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/Koushik0901/Swift-SRGAN/releases/download/v0.1/swift_srgan_2x.pth.tar",
        name="swift_srgan_2x.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/Koushik0901/Swift-SRGAN/releases/download/v0.1/swift_srgan_4x.pth.tar",
        name="swift_srgan_4x.pth",
    )
    assert_size_requirements(file.load_model())


def test_SwiftSRGAN_2x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Koushik0901/Swift-SRGAN/releases/download/v0.1/swift_srgan_2x.pth.tar",
        name="swift_srgan_2x.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwiftSRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_SwiftSRGAN_4x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Koushik0901/Swift-SRGAN/releases/download/v0.1/swift_srgan_4x.pth.tar",
        name="swift_srgan_4x.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwiftSRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
