from spandrel import ModelLoader
from spandrel.architectures.ESRGAN import RRDBNet

from .util import ImageTestNames, ModelFile, compare_images_to_results, disallowed_props


def test_ESRGAN_community(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-Anti-Aliasing.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )


def test_BSRGAN(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )


def test_BSRGAN_2x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGANx2.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )


def test_RealSR_DPED(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/RealSR_DPED.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )


def test_RealSR_JPEG(snapshot):
    file = ModelFile.from_url(
        "https://github.com/cszn/KAIR/releases/download/v1.0/RealSR_JPEG.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )


def test_RealESRGAN_x4plus(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )


def test_RealESRGAN_x2plus(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )


def test_RealESRGAN_x4plus_anime_6B(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )


def test_RealESRNet_x4plus(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RRDBNet)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )
