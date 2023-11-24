from spandrel.architectures.GRLIR import GRLIR, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
)


def test_GRLIR_load():
    assert_loads_correctly(
        load,
        lambda: GRLIR(),
        lambda: GRLIR(in_channels=1, out_channels=3),
        lambda: GRLIR(in_channels=4, out_channels=4),
        lambda: GRLIR(embed_dim=16),
        # embed_dim=16 makes tests go faster
        lambda: GRLIR(embed_dim=16, upsampler="pixelshuffle", upscale=2),
        lambda: GRLIR(embed_dim=16, upsampler="pixelshuffle", upscale=3),
        lambda: GRLIR(embed_dim=16, upsampler="pixelshuffle", upscale=4),
        lambda: GRLIR(embed_dim=16, upsampler="pixelshuffle", upscale=8),
        lambda: GRLIR(embed_dim=16, upsampler="pixelshuffledirect", upscale=2),
        lambda: GRLIR(embed_dim=16, upsampler="pixelshuffledirect", upscale=3),
        lambda: GRLIR(embed_dim=16, upsampler="pixelshuffledirect", upscale=4),
        lambda: GRLIR(embed_dim=16, upsampler="pixelshuffledirect", upscale=8),
        lambda: GRLIR(embed_dim=16, upsampler="nearest+conv", upscale=4),
        lambda: GRLIR(
            embed_dim=16,
            depths=[4, 5, 3, 2, 1],
            num_heads_window=[2, 3, 5, 1, 3],
            num_heads_stripe=[2, 4, 7, 1, 1],
        ),
        lambda: GRLIR(mlp_ratio=2),
        lambda: GRLIR(mlp_ratio=3),
        lambda: GRLIR(qkv_proj_type="linear", qkv_bias=True),
        lambda: GRLIR(qkv_proj_type="linear", qkv_bias=False),
        lambda: GRLIR(qkv_proj_type="separable_conv", qkv_bias=True),
        lambda: GRLIR(qkv_proj_type="separable_conv", qkv_bias=False),
        lambda: GRLIR(conv_type="1conv"),
        lambda: GRLIR(conv_type="1conv1x1"),
        lambda: GRLIR(conv_type="linear"),
        lambda: GRLIR(conv_type="3conv"),
        # some actual training configs
        lambda: GRLIR(
            upscale=4,
            img_size=64,
            window_size=8,
            depths=[4, 4, 4, 4],
            embed_dim=32,
            num_heads_window=[2, 2, 2, 2],
            num_heads_stripe=[2, 2, 2, 2],
            mlp_ratio=2,
            qkv_proj_type="linear",
            anchor_proj_type="avgpool",
            anchor_window_down_factor=2,
            out_proj_type="linear",
            conv_type="1conv",
            upsampler="pixelshuffledirect",
        ),
        lambda: GRLIR(
            upscale=4,
            img_size=64,
            window_size=8,
            depths=[4, 4, 4, 4, 4, 4, 4, 4],
            embed_dim=192,
            num_heads_window=[4, 4, 4, 4, 4, 4, 4, 4],
            num_heads_stripe=[4, 4, 4, 4, 4, 4, 4, 4],
            mlp_ratio=2,
            qkv_proj_type="linear",
            anchor_proj_type="avgpool",
            anchor_window_down_factor=2,
            out_proj_type="linear",
            conv_type="1conv",
            upsampler="pixelshuffle",
        ),
        lambda: GRLIR(
            upscale=4,
            img_size=64,
            window_size=8,
            depths=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            embed_dim=256,
            num_heads_window=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            num_heads_stripe=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            mlp_ratio=4,
            qkv_proj_type="linear",
            anchor_proj_type="avgpool",
            anchor_window_down_factor=2,
            out_proj_type="linear",
            conv_type="1conv",
            upsampler="pixelshuffle",
        ),
        condition=lambda a, b: (
            a.in_channels == b.in_channels
            and a.out_channels == b.out_channels
            and a.embed_dim == b.embed_dim
            and a.upsampler == b.upsampler
            # upscale is only defined if we have an upsampler
            and (not a.upsampler or a.upscale == b.upscale)
            and a.pad_size == b.pad_size
            and a.input_resolution == b.input_resolution
            and a.window_size == b.window_size
            and a.shift_size == b.shift_size
            and a.stripe_size == b.stripe_size
            and a.stripe_groups == b.stripe_groups
            and a.pretrained_window_size == b.pretrained_window_size
            and a.pretrained_stripe_size == b.pretrained_stripe_size
            # anchor_window_down_factor isn't well-defined
            # and a.anchor_window_down_factor == b.anchor_window_down_factor
        ),
    )


def test_GRLIR_dn_grl_tiny_c1(snapshot):
    file = ModelFile.from_url(
        "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/dn_grl_tiny_c1.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRLIR)
    # this model is weird, so no inference test


def test_GRLIR_dn_grl_base_c1s25(snapshot):
    file = ModelFile.from_url(
        "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/dn_grl_base_c1s25.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRLIR)
    # we don't have grayscale images yet


def test_GRLIR_jpeg_grl_small_c1q30(snapshot):
    file = ModelFile.from_url(
        "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/jpeg_grl_small_c1q30.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRLIR)
    # we don't have grayscale images yet


def test_GRLIR_dn_grl_small_c3s15(snapshot):
    file = ModelFile.from_url(
        "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/dn_grl_small_c3s15.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRLIR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_GRLIR_dn_grl_base_c3s50(snapshot):
    file = ModelFile.from_url(
        "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/dn_grl_base_c3s50.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRLIR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_GRLIR_db_motion_grl_base_gopro(snapshot):
    file = ModelFile.from_url(
        "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/db_motion_grl_base_gopro.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRLIR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_GRLIR_jpeg_grl_small_c3(snapshot):
    file = ModelFile.from_url(
        "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/jpeg_grl_small_c3.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRLIR)
    # this model is weird, so no inference test


def test_GRLIR_jpeg_grl_small_c3q20(snapshot):
    file = ModelFile.from_url(
        "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/jpeg_grl_small_c3q20.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRLIR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_64, TestImage.JPEG_15],
    )


def test_GRLIR_sr_grl_tiny_c3x3(snapshot):
    file = ModelFile.from_url(
        "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/sr_grl_tiny_c3x3.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRLIR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_GRLIR_sr_grl_tiny_c3x4(snapshot):
    file = ModelFile.from_url(
        "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/sr_grl_tiny_c3x4.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRLIR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
