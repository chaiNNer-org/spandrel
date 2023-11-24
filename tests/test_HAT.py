from spandrel.architectures.HAT import HAT, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
)


def test_HAT_load():
    assert_loads_correctly(
        load,
        lambda: HAT(),
        lambda: HAT(in_chans=1),
        lambda: HAT(in_chans=4),
        lambda: HAT(embed_dim=64),
        lambda: HAT(depths=(1, 2, 3, 2, 1), num_heads=(5, 4, 3, 5, 6)),
        lambda: HAT(window_size=13),
        lambda: HAT(compress_ratio=2),
        lambda: HAT(squeeze_factor=15),
        lambda: HAT(overlap_ratio=0.75),
        lambda: HAT(mlp_ratio=3),
        lambda: HAT(qkv_bias=False),
        lambda: HAT(ape=True),
        lambda: HAT(patch_norm=False),
        lambda: HAT(resi_connection="1conv"),
        lambda: HAT(resi_connection="identity"),
        lambda: HAT(upsampler="pixelshuffle", upscale=1),
        lambda: HAT(upsampler="pixelshuffle", upscale=2),
        lambda: HAT(upsampler="pixelshuffle", upscale=3),
        lambda: HAT(upsampler="pixelshuffle", upscale=4),
        lambda: HAT(upsampler="pixelshuffle", upscale=8),
        lambda: HAT(upsampler="pixelshuffle", num_feat=32),
        condition=lambda a, b: (
            a.window_size == b.window_size
            and a.overlap_ratio == b.overlap_ratio
            and a.upsampler == b.upsampler
            # upscale is only defined if we have an upsampler
            and (not a.upsampler or a.upscale == b.upscale)
            and a.num_layers == b.num_layers
            and a.embed_dim == b.embed_dim
            and a.ape == b.ape
            and a.patch_norm == b.patch_norm
            and a.mlp_ratio == b.mlp_ratio
        ),
    )


def test_HAT_community1(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Phhofm/models/raw/main/4xLexicaHAT/4xLexicaHAT.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_community2(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Phhofm/models/raw/main/4xNomos8kSCHAT-S/4xNomos8kSCHAT-S.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
