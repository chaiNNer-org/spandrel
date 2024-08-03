from spandrel.architectures.sudo_SPANPlus import sudo_SPANPlus

from .util import assert_loads_correctly


from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)



def test_ARCH_model_name(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/secret_sauce.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
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