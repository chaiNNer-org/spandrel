from spandrel.architectures.ESRGAN import ESRGAN, ESRGANArch

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
        ESRGANArch(),
        lambda: ESRGAN(in_nc=3, out_nc=3, num_filters=64, num_blocks=23, scale=4),
        lambda: ESRGAN(in_nc=1, out_nc=3, num_filters=32, num_blocks=11, scale=2),
        lambda: ESRGAN(in_nc=1, out_nc=1, num_filters=64, num_blocks=23, scale=1),
        lambda: ESRGAN(in_nc=4, out_nc=4, num_filters=64, num_blocks=23, scale=8),
        lambda: ESRGAN(scale=4, plus=True),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-NMKD-YandereNeo.pth"
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-Anti-Aliasing.pth"
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    )
    assert_size_requirements(file.load_model())


def test_ESRGAN_community(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-Anti-Aliasing.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ESRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_ESRGAN_community_2x(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-BIGOLDIES.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ESRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_8, TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_ESRGAN_community_4x(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-NMKD-YandereNeo.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ESRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_8, TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_ESRGAN_community_8x(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/8x-ESRGAN.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ESRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_8, TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_BSRGAN(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ESRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_BSRGAN_2x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGANx2.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ESRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealSR_DPED(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/RealSR_DPED.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ESRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealSR_JPEG(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/RealSR_JPEG.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ESRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealESRGAN_x4plus(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ESRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealESRGAN_x2plus(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ESRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealESRGAN_x4plus_anime_6B(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ESRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealESRNet_x4plus(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ESRGAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
