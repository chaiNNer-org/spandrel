from spandrel.architectures.RGT import RGT, RGTArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
)


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
        "https://drive.google.com/file/d/1uSgjg5ipivuEz6AlS6jhEH3oF6tuudUb/view?usp=drive_link",
        name="RGT_x2.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1F_mr-bjYtP5FQLEEo_MOthj9Bk6kQLgu/view?usp=drive_link",
        name="RGT_x3.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1uULArtV1EcPbS3ujVbZvJIRMogldRMHr/view?usp=drive_link",
        name="RGT_x4.pth",
    )
    assert_size_requirements(file.load_model())


def test_RGT_x2(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1uSgjg5ipivuEz6AlS6jhEH3oF6tuudUb/view?usp=drive_link",
        name="RGT_x2.pth",
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
        "https://drive.google.com/file/d/1F_mr-bjYtP5FQLEEo_MOthj9Bk6kQLgu/view?usp=drive_link",
        name="RGT_x3.pth",
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
        "https://drive.google.com/file/d/1uULArtV1EcPbS3ujVbZvJIRMogldRMHr/view?usp=drive_link",
        name="RGT_x4.pth",
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
        "https://drive.google.com/file/d/1NNaj3UH5smEylwVabQLMEdey-FoaqnM_/view?usp=drive_link",
        name="RGT_S_x4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RGT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
