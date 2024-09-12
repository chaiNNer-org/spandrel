from spandrel.architectures.sudo_SPANPlus import sudo_SPANPlus

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_sudo_SPANPlus(snapshot):
    file = ModelFile.from_url("secret_sauce.pth")
    model = file.load_model()
    # assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, sudo_SPANPlus)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_load():
    assert_loads_correctly(
        lambda: sudo_SPANPlus(),
    )
