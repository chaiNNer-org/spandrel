from spandrel.architectures.DITN import DITN, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
)


def test_DITN_load():
    assert_loads_correctly(
        load,
        lambda: DITN(),
        lambda: DITN(inp_channels=4),
        lambda: DITN(inp_channels=1),
        lambda: DITN(dim=4),
        lambda: DITN(dim=16),
        lambda: DITN(UFONE_blocks=1),
        lambda: DITN(UFONE_blocks=2),
        lambda: DITN(UFONE_blocks=5),
        lambda: DITN(ITL_blocks=7, SAL_blocks=3),
        lambda: DITN(ffn_expansion_factor=3),
        lambda: DITN(bias=True),
        lambda: DITN(upscale=6),
        condition=lambda a, b: (
            a.patch_size == b.patch_size
            and a.dim == b.dim
            and a.scale == b.scale
            and a.SAL_blocks == b.SAL_blocks
            and a.ITL_blocks == b.ITL_blocks
        ),
    )


def test_DITN_Real_GAN_x4(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/12y6WjNowBkJ982fMql_yj6zBpwPKuhV2/view?usp=drive_link",
        name="DITN_Real_GAN_x4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DITN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_DITN_Real_x4(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1V9KHVPvLtOTwYINaD8_nGRuc37KZurkJ/view?usp=drive_link",
        name="DITN_Real_x4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DITN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
