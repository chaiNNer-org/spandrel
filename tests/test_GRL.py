from spandrel.architectures.GRL import GRL, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
)


def test_GRL_load():
    assert_loads_correctly(
        load,
        lambda: GRL(),
        lambda: GRL(in_channels=1, out_channels=3),
        lambda: GRL(in_channels=4, out_channels=4),
        lambda: GRL(embed_dim=16),
        # embed_dim=16 makes tests go faster
        lambda: GRL(embed_dim=16, upsampler="pixelshuffle", upscale=2),
        lambda: GRL(embed_dim=16, upsampler="pixelshuffle", upscale=3),
        lambda: GRL(embed_dim=16, upsampler="pixelshuffle", upscale=4),
        lambda: GRL(embed_dim=16, upsampler="pixelshuffle", upscale=8),
        lambda: GRL(embed_dim=16, upsampler="pixelshuffledirect", upscale=2),
        lambda: GRL(embed_dim=16, upsampler="pixelshuffledirect", upscale=3),
        lambda: GRL(embed_dim=16, upsampler="pixelshuffledirect", upscale=4),
        lambda: GRL(embed_dim=16, upsampler="pixelshuffledirect", upscale=8),
        lambda: GRL(embed_dim=16, upsampler="nearest+conv", upscale=4),
        lambda: GRL(
            embed_dim=16,
            depths=[4, 5, 3, 2, 1],
            num_heads_window=[2, 3, 5, 1, 3],
            num_heads_stripe=[2, 4, 7, 1, 1],
        ),
        lambda: GRL(mlp_ratio=2),
        lambda: GRL(mlp_ratio=3),
        lambda: GRL(qkv_proj_type="linear", qkv_bias=True),
        lambda: GRL(qkv_proj_type="linear", qkv_bias=False),
        lambda: GRL(qkv_proj_type="separable_conv", qkv_bias=True),
        lambda: GRL(qkv_proj_type="separable_conv", qkv_bias=False),
        lambda: GRL(conv_type="1conv"),
        lambda: GRL(conv_type="1conv1x1"),
        lambda: GRL(conv_type="linear"),
        lambda: GRL(conv_type="3conv"),
        # These require non-persistent buffers to be detected
        # lambda: GRL(
        #     window_size=16,
        #     stripe_size=[32, 64],
        #     anchor_window_down_factor=1,
        # ),
        # lambda: GRL(
        #     window_size=16,
        #     stripe_size=[32, 64],
        #     anchor_window_down_factor=2,
        # ),
        # lambda: GRL(
        #     window_size=16,
        #     stripe_size=[32, 64],
        #     anchor_window_down_factor=4,
        # ),
        # some actual training configs
        lambda: GRL(
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
        lambda: GRL(
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
        lambda: GRL(
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
            and a.input_resolution == b.input_resolution
            # those aren't supported right now
            # and a.pad_size == b.pad_size
            # and a.window_size == b.window_size
            # and a.stripe_size == b.stripe_size
            # and a.shift_size == b.shift_size
            and a.stripe_groups == b.stripe_groups
            and a.pretrained_window_size == b.pretrained_window_size
            and a.pretrained_stripe_size == b.pretrained_stripe_size
            # and a.anchor_window_down_factor == b.anchor_window_down_factor
        ),
    )


# def test_GRL_dn_grl_tiny_c1(snapshot):
#     file = ModelFile.from_url(
#         "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/dn_grl_tiny_c1.ckpt"
#     )
#     model = file.load_model()
#     assert model == snapshot(exclude=disallowed_props)
#     assert isinstance(model.model, GRL)
#     # this model is weird, so no inference test


# def test_GRL_dn_grl_base_c1s25(snapshot):
#     file = ModelFile.from_url(
#         "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/dn_grl_base_c1s25.ckpt"
#     )
#     model = file.load_model()
#     assert model == snapshot(exclude=disallowed_props)
#     assert isinstance(model.model, GRL)
#     # we don't have grayscale images yet


# def test_GRL_jpeg_grl_small_c1q30(snapshot):
#     file = ModelFile.from_url(
#         "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/jpeg_grl_small_c1q30.ckpt"
#     )
#     model = file.load_model()
#     assert model == snapshot(exclude=disallowed_props)
#     assert isinstance(model.model, GRL)
#     # we don't have grayscale images yet


# def test_GRL_dn_grl_small_c3s15(snapshot):
#     file = ModelFile.from_url(
#         "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/dn_grl_small_c3s15.ckpt"
#     )
#     model = file.load_model()
#     assert model == snapshot(exclude=disallowed_props)
#     assert isinstance(model.model, GRL)
#     assert_image_inference(
#         file,
#         model,
#         [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
#     )


# def test_GRL_dn_grl_base_c3s50(snapshot):
#     file = ModelFile.from_url(
#         "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/dn_grl_base_c3s50.ckpt"
#     )
#     model = file.load_model()
#     assert model == snapshot(exclude=disallowed_props)
#     assert isinstance(model.model, GRL)
#     assert_image_inference(
#         file,
#         model,
#         [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
#     )


# def test_GRL_db_motion_grl_base_gopro(snapshot):
#     file = ModelFile.from_url(
#         "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/db_motion_grl_base_gopro.ckpt"
#     )
#     model = file.load_model()
#     assert model == snapshot(exclude=disallowed_props)
#     assert isinstance(model.model, GRL)
#     assert_image_inference(
#         file,
#         model,
#         [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
#     )


# def test_GRL_jpeg_grl_small_c3(snapshot):
#     file = ModelFile.from_url(
#         "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/jpeg_grl_small_c3.ckpt"
#     )
#     model = file.load_model()
#     assert model == snapshot(exclude=disallowed_props)
#     assert isinstance(model.model, GRL)
#     # this model is weird, so no inference test


# def test_GRL_jpeg_grl_small_c3q20(snapshot):
#     file = ModelFile.from_url(
#         "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/jpeg_grl_small_c3q20.ckpt"
#     )
#     model = file.load_model()
#     assert model == snapshot(exclude=disallowed_props)
#     assert isinstance(model.model, GRL)
#     assert_image_inference(
#         file,
#         model,
#         [TestImage.SR_64, TestImage.JPEG_15],
#     )


def test_GRL_sr_grl_tiny_c3x3(snapshot):
    file = ModelFile.from_url(
        "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/sr_grl_tiny_c3x3.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRL)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_GRL_sr_grl_tiny_c3x4(snapshot):
    file = ModelFile.from_url(
        "https://github.com/ofsoundof/GRL-Image-Restoration/releases/download/v1.0.0/sr_grl_tiny_c3x4.ckpt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRL)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_GRL_bsr_grl_base(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1JdzeTFiBVSia7PmSvr5VduwDdLnirxAG/view",
        name="bsr_grl_base.safetensors",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, GRL)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
