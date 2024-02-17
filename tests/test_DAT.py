from spandrel import ModelLoader
from spandrel.architectures.DAT import DAT, DATArch

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
        DATArch(),
        lambda: DAT(),
        lambda: DAT(embed_dim=60),
        lambda: DAT(in_chans=1),
        lambda: DAT(in_chans=4),
        lambda: DAT(depth=[2, 3], num_heads=[2, 5]),
        lambda: DAT(depth=[2, 3, 4, 2], num_heads=[2, 3, 2, 2]),
        lambda: DAT(depth=[2, 3, 4, 2, 5], num_heads=[2, 3, 2, 2, 3]),
        lambda: DAT(upsampler="pixelshuffle", upscale=1),
        lambda: DAT(upsampler="pixelshuffle", upscale=2),
        lambda: DAT(upsampler="pixelshuffle", upscale=3),
        lambda: DAT(upsampler="pixelshuffle", upscale=4),
        lambda: DAT(upsampler="pixelshuffle", upscale=8),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=1),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=2),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=3),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=4),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=8),
        lambda: DAT(resi_connection="3conv"),
        lambda: DAT(qkv_bias=False),
        lambda: DAT(split_size=[4, 4]),
        lambda: DAT(split_size=[2, 8]),
        condition=lambda a, b: (
            a.num_layers == b.num_layers
            and a.upscale == b.upscale
            and a.upsampler == b.upsampler
            and a.embed_dim == b.embed_dim
            and a.num_features == b.num_features
        ),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1iY30DyLYjar-2DjrJtAv2chCOlw4xiOj/view",
        name="DAT_S_x4.pth",
    )
    assert_size_requirements(file.load_model())


def test_DAT_S_x4(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1iY30DyLYjar-2DjrJtAv2chCOlw4xiOj/view",
        name="DAT_S_x4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_DAT_S_x3(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1Fmj7VFKznbak-atd6pEu59UTZxKTXYVi/view",
        name="DAT_S_x3.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_DAT_x4(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1pEhXmg--IWHaZOwHUFdh7TEJqt2qeuYg/view",
        name="DAT_x4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_DAT_2_x4(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1sfB15jklXRjGiZZWgYXZAYc2Ut4TuKOz/view",
        name="DAT_2_x4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_DAT_tags(snapshot):
    loader = ModelLoader()

    # https://github.com/muslll/neosr/blob/master/neosr/archs/dat_arch.py#L876

    dat_light = DAT(
        in_chans=3,
        img_range=1.0,
        depth=[18],
        embed_dim=60,
        num_heads=[6],
        expansion_factor=2,
        resi_connection="3conv",
        split_size=[8, 32],
        upsampler="pixelshuffledirect",
    )
    assert loader.load_from_state_dict(dat_light.state_dict()) == snapshot(
        exclude=disallowed_props
    )

    dat_small = DAT(
        in_chans=3,
        img_range=1.0,
        split_size=[8, 16],
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        expansion_factor=2,
        resi_connection="1conv",
    )
    assert loader.load_from_state_dict(dat_small.state_dict()) == snapshot(
        exclude=disallowed_props
    )

    dat_medium = DAT(
        in_chans=3,
        img_range=1.0,
        split_size=[8, 32],
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        expansion_factor=4,
        resi_connection="1conv",
    )
    assert loader.load_from_state_dict(dat_medium.state_dict()) == snapshot(
        exclude=disallowed_props
    )

    dat_2 = DAT(
        in_chans=3,
        img_range=1.0,
        split_size=[8, 32],
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        expansion_factor=2,
        resi_connection="1conv",
    )
    assert loader.load_from_state_dict(dat_2.state_dict()) == snapshot(
        exclude=disallowed_props
    )
