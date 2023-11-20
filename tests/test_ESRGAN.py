from spandrel import ModelLoader
from spandrel.architectures.ESRGAN import RRDBNet

from .util import ModelFile, TestImage, assert_image_inference, disallowed_props


def test_ESRGAN_community(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-Anti-Aliasing.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_ESRGAN_community_2x(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-BIGOLDIES.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_8, TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_ESRGAN_community_4x(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-NMKD-YandereNeo.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_8, TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_ESRGAN_community_8x(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/8x-ESRGAN.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_8, TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_BSRGAN(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_BSRGAN_2x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGANx2.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealSR_DPED(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/RealSR_DPED.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealSR_JPEG(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/RealSR_JPEG.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealESRGAN_x4plus(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealESRGAN_x2plus(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealESRGAN_x4plus_anime_6B(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RealESRNet_x4plus(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
