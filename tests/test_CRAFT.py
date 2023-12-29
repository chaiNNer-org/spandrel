from spandrel.architectures.CRAFT import CRAFT, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
)


def test_CRAFT_load():
    assert_loads_correctly(
        load,
        lambda: CRAFT(),
        lambda: CRAFT(embed_dim=60),
        lambda: CRAFT(mlp_ratio=4.0),
        lambda: CRAFT(in_chans=1),
        lambda: CRAFT(in_chans=4),
        lambda: CRAFT(depths=[2, 2], num_heads=[6, 6]),
        lambda: CRAFT(depths=[2, 2, 2], num_heads=[6, 6, 6]),
        lambda: CRAFT(depths=[2, 2, 2, 2], num_heads=[6, 6, 6, 6]),
        lambda: CRAFT(upscale=1),
        lambda: CRAFT(upscale=2),
        lambda: CRAFT(upscale=3),
        lambda: CRAFT(upscale=4),
        condition=lambda a, b: (
            a.num_layers == b.num_layers
            and a.upscale == b.upscale
            and a.embed_dim == b.embed_dim
        ),
    )


def test_CRAFT_x2(snapshot):
    file = ModelFile.from_url_zip(
        "https://drive.google.com/file/d/13wAmc93BPeBUBQ24zUZOuUpdBFG2aAY5/view",
        rel_model_path="pretrained_models/CRAFT_MODEL_X2.pth",
        name="CRAFT_MODEL_X2.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, CRAFT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_CRAFT_x3(snapshot):
    file = ModelFile.from_url_zip(
        "https://drive.google.com/file/d/13wAmc93BPeBUBQ24zUZOuUpdBFG2aAY5/view",
        rel_model_path="pretrained_models/CRAFT_MODEL_X3.pth",
        name="CRAFT_MODEL_X3.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, CRAFT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_CRAFT_x4(snapshot):
    file = ModelFile.from_url_zip(
        "https://drive.google.com/file/d/13wAmc93BPeBUBQ24zUZOuUpdBFG2aAY5/view",
        rel_model_path="pretrained_models/CRAFT_MODEL_X4.pth",
        name="CRAFT_MODEL_X4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, CRAFT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
