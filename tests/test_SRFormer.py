from spandrel_extra_arches.architectures.SRFormer import SRFormer, SRFormerArch

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
        SRFormerArch(),
        lambda: SRFormer(window_size=8),
        lambda: SRFormer(window_size=8, in_chans=1),
        lambda: SRFormer(window_size=8, in_chans=4),
        lambda: SRFormer(window_size=8, embed_dim=64),
        lambda: SRFormer(
            window_size=8, depths=(3, 4, 5, 1, 3), num_heads=(2, 3, 1, 1, 1)
        ),
        lambda: SRFormer(window_size=8, mlp_ratio=3),
        lambda: SRFormer(window_size=8, resi_connection="1conv"),
        lambda: SRFormer(window_size=8, resi_connection="3conv"),
        lambda: SRFormer(window_size=8, patch_norm=False),
        lambda: SRFormer(window_size=8, qkv_bias=False),
        lambda: SRFormer(window_size=8, ape=True),
        lambda: SRFormer(window_size=8, upscale=1, upsampler=""),
        lambda: SRFormer(window_size=8, upscale=1, upsampler="pixelshuffle"),
        lambda: SRFormer(window_size=8, upscale=2, upsampler="pixelshuffle"),
        lambda: SRFormer(window_size=8, upscale=3, upsampler="pixelshuffle"),
        lambda: SRFormer(window_size=8, upscale=4, upsampler="pixelshuffle"),
        lambda: SRFormer(window_size=8, upscale=1, upsampler="pixelshuffledirect"),
        lambda: SRFormer(window_size=8, upscale=2, upsampler="pixelshuffledirect"),
        lambda: SRFormer(window_size=8, upscale=3, upsampler="pixelshuffledirect"),
        lambda: SRFormer(window_size=8, upscale=4, upsampler="pixelshuffledirect"),
        lambda: SRFormer(window_size=8, upscale=4, upsampler="nearest+conv"),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1lU8SsKeaTwBSC5bP69LjBuJs69Qt4Rsf/view?usp=drive_link",
        name="SRFormer_SRx2_DF2K.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1Eeei_NEjDeni7ysSejmR7AwyG24fO5bp/view?usp=drive_link",
        name="SRFormerLight_SRx3_DIV2K.pth",
    )
    assert_size_requirements(file.load_model())


def test_SRFormer_SRx2_DF2K(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1lU8SsKeaTwBSC5bP69LjBuJs69Qt4Rsf/view?usp=drive_link",
        name="SRFormer_SRx2_DF2K.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SRFormer)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_SRFormer_SRx4_DF2K(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/13_fpD4aDE1wbEYX8yGWA3mVLZOCRWkWv/view?usp=drive_link",
        name="SRFormer_SRx4_DF2K.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SRFormer)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_SRFormerLight_SRx3_DIV2K(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1Eeei_NEjDeni7ysSejmR7AwyG24fO5bp/view?usp=drive_link",
        name="SRFormerLight_SRx3_DIV2K.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SRFormer)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
