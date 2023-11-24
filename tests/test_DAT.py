from spandrel import ModelLoader
from spandrel.architectures.DAT import DAT, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
)


def test_DAT_load():
    assert_loads_correctly(
        load,
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


def test_DAT_S_x4(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1iY30DyLYjar-2DjrJtAv2chCOlw4xiOj/view",
        name="DAT_S_x4.pth",
    )
    model = ModelLoader().load_from_file(file.path)
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
    model = ModelLoader().load_from_file(file.path)
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
    model = ModelLoader().load_from_file(file.path)
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
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
