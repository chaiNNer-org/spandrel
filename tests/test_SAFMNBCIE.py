from spandrel.architectures.SAFMNBCIE import SAFMN_BCIE, SAFMNBCIEArch

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
        SAFMNBCIEArch(),
        lambda: SAFMN_BCIE(
            dim=36, n_blocks=8, num_layers=1, ffn_scale=2.0, upscaling_factor=4
        ),
        lambda: SAFMN_BCIE(
            dim=36, n_blocks=8, num_layers=3, ffn_scale=3.0, upscaling_factor=3
        ),
        lambda: SAFMN_BCIE(
            dim=8, n_blocks=3, num_layers=4, ffn_scale=5.0, upscaling_factor=2
        ),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/sunny2109/SAFMN/releases/download/v0.1.1/SAFMN_BCIE.pth",
    )
    assert_size_requirements(file.load_model())


def test_SAFMN_BCIE(snapshot):
    file = ModelFile.from_url(
        "https://github.com/sunny2109/SAFMN/releases/download/v0.1.1/SAFMN_BCIE.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SAFMN_BCIE)
    assert_image_inference(file, model, [TestImage.JPEG_15])
