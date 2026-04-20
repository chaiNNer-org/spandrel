from spandrel.architectures.IPT import IPT, IPTArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    assert_training,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        IPTArch(),
        lambda: IPT(),
        lambda: IPT(patch_size=24, n_feats=16),
        lambda: IPT(rgb_range=1),
        lambda: IPT(scale=[1, 2, 3, 4, 8, 16]),
        lambda: IPT(num_layers=6),
        lambda: IPT(num_queries=3),
        lambda: IPT(mlp=False),
        lambda: IPT(no_pos=True),
        lambda: IPT(no_norm=True),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1KFnYnSxXnXwmEB80pM79qi8Pr_vqkWYZ/view?usp=drive_link",
        name="IPT_denoise50.pth",
    )
    assert_size_requirements(file.load_model(), max_candidates=4)


def test_train():
    assert_training(IPTArch(), IPT())


def test_IPT_denoise50(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1KFnYnSxXnXwmEB80pM79qi8Pr_vqkWYZ/view?usp=drive_link",
        name="IPT_denoise50.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, IPT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_32],
    )
