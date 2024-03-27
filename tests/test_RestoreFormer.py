from spandrel.architectures.RestoreFormer import RestoreFormer, RestoreFormerArch

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
        RestoreFormerArch(),
        lambda: RestoreFormer(),
        lambda: RestoreFormer(n_embed=256, embed_dim=32),
        lambda: RestoreFormer(ch=32),
        lambda: RestoreFormer(in_channels=1, out_ch=1),
        lambda: RestoreFormer(in_channels=1, out_ch=3),
        lambda: RestoreFormer(in_channels=4, out_ch=4),
        lambda: RestoreFormer(num_res_blocks=3),
        lambda: RestoreFormer(ch_mult=(1, 3, 6)),
        lambda: RestoreFormer(z_channels=64, double_z=True),
        lambda: RestoreFormer(enable_mid=False),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    )
    assert_size_requirements(file.load_model(), max_size=128)


def test_RestoreFormer(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RestoreFormer)
    assert_image_inference(file, model, [TestImage.LR_FACE])
