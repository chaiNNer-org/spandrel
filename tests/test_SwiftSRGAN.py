from spandrel import ModelLoader
from spandrel.architectures.SwiftSRGAN import SwiftSRGAN

from .util import ModelFile, compare_images_to_results, disallowed_props


def test_SwiftSRGan_2x(snapshot):
    file = ModelFile("swift_srgan_2x.pth").download(
        "https://github.com/Koushik0901/Swift-SRGAN/releases/download/v0.1/swift_srgan_2x.pth.tar"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwiftSRGAN)
    assert compare_images_to_results(file.name, model.model)


def test_SwiftSRGan_4x(snapshot):
    file = ModelFile("swift_srgan_4x.pth").download(
        "https://github.com/Koushik0901/Swift-SRGAN/releases/download/v0.1/swift_srgan_4x.pth.tar"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SwiftSRGAN)
    assert compare_images_to_results(file.name, model.model)
