from spandrel.architectures.RCAN import RCAN, RCANArch

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
        RCANArch(),
        lambda: RCAN(),
        lambda: RCAN(scale=1),
        lambda: RCAN(scale=2),
        lambda: RCAN(scale=3),
        lambda: RCAN(scale=8),
        lambda: RCAN(n_resgroups=5),
        lambda: RCAN(n_resblocks=10),
        lambda: RCAN(n_feats=32),
        lambda: RCAN(n_colors=1),
        lambda: RCAN(norm=False),
        lambda: RCAN(kernel_size=7),
        lambda: RCAN(reduction=8),
        lambda: RCAN(scale=1, unshuffle_mod=True),
        lambda: RCAN(scale=1, unshuffle_mod=True, n_feats=32),
        lambda: RCAN(scale=2, unshuffle_mod=True),
        lambda: RCAN(scale=2, unshuffle_mod=True, n_colors=1),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1CZaPYhO1EQRodctgPWKeC2yKbI6nX2-B/view?usp=sharing",
        name="RCAN_BIX4.safetensors",
    )
    assert_size_requirements(file.load_model())


def test_rcan_bix4(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1CZaPYhO1EQRodctgPWKeC2yKbI6nX2-B/view?usp=sharing",
        name="RCAN_BIX4.safetensors",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RCAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
