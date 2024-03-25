from spandrel.architectures.RGT import RGT, RGTArch

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
        RGTArch(),
        lambda: RGT(),
        lambda: RGT(in_chans=4, embed_dim=90),
        lambda: RGT(depth=[5, 6, 2, 3, 9], num_heads=[2, 6, 2, 9, 4]),
        lambda: RGT(mlp_ratio=3.0, qkv_bias=False),
        lambda: RGT(resi_connection="1conv"),
        lambda: RGT(resi_connection="3conv"),
        lambda: RGT(c_ratio=0.75),
        lambda: RGT(split_size=[16, 16]),
        lambda: RGT(split_size=[4, 4]),
        lambda: RGT(split_size=[8, 32]),
        lambda: RGT(upscale=1),
        lambda: RGT(upscale=2),
        lambda: RGT(upscale=3),
        lambda: RGT(upscale=4),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/rgt/2x-RGT.pth"
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/rgt/3x-RGT.pth"
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/rgt/4x-RGT.pth"
    )
    assert_size_requirements(file.load_model())


def test_RGT_x2(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/rgt/2x-RGT.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RGT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RGT_x3(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/rgt/3x-RGT.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RGT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RGT_x4(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/rgt/4x-RGT.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RGT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_RGT_S_x4(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/rgt/4x-RGT_S.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RGT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
