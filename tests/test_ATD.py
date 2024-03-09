from spandrel.architectures.ATD import ATD, ATDArch

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
        ATDArch(),
        lambda: ATD(),
        lambda: ATD(in_chans=4, embed_dim=60),
        lambda: ATD(window_size=4),
        lambda: ATD(depths=(4, 6, 8, 7, 5), num_heads=(4, 6, 8, 12, 5)),
        lambda: ATD(num_tokens=32, reducted_dim=3, convffn_kernel_size=7, mlp_ratio=3),
        lambda: ATD(qkv_bias=False),
        lambda: ATD(patch_norm=False),
        lambda: ATD(ape=True),
        lambda: ATD(resi_connection="1conv"),
        lambda: ATD(resi_connection="3conv"),
        lambda: ATD(upsampler="", upscale=1),
        lambda: ATD(upsampler="nearest+conv", upscale=4),
        lambda: ATD(upsampler="pixelshuffle", upscale=1),
        lambda: ATD(upsampler="pixelshuffle", upscale=2),
        lambda: ATD(upsampler="pixelshuffle", upscale=3),
        lambda: ATD(upsampler="pixelshuffle", upscale=4),
        lambda: ATD(upsampler="pixelshuffle", upscale=8),
        lambda: ATD(upsampler="pixelshuffledirect", upscale=1),
        lambda: ATD(upsampler="pixelshuffledirect", upscale=2),
        lambda: ATD(upsampler="pixelshuffledirect", upscale=3),
        lambda: ATD(upsampler="pixelshuffledirect", upscale=4),
        lambda: ATD(upsampler="pixelshuffledirect", upscale=8),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1ZxK7gMJXgeyHgeOaKbzpXtoElDmRWKkU/view?usp=drive_link",
        name="101_ATD_light_SRx2_scratch.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1CxzxLWMIHDHip2nDF6A2sLBTkpypI8GU/view?usp=drive_link",
        name="103_ATD_light_SRx4_finetune.pth",
    )
    assert_size_requirements(file.load_model())


def test_101_ATD_light_SRx2_scratch(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1ZxK7gMJXgeyHgeOaKbzpXtoElDmRWKkU/view?usp=drive_link",
        name="101_ATD_light_SRx2_scratch.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ATD)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
        tolerance=3,
    )


def test_102_ATD_light_SRx3_finetune(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1mPcWMyDO9lOUD6DrrmNxFR67sGqnOJxp/view?usp=drive_link",
        name="102_ATD_light_SRx3_finetune.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ATD)


def test_103_ATD_light_SRx4_finetune(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1CxzxLWMIHDHip2nDF6A2sLBTkpypI8GU/view?usp=drive_link",
        name="103_ATD_light_SRx4_finetune.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ATD)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
        tolerance=3,
    )


def test_003_ATD_SRx4_finetune(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1J9kR9OyrOxtJ5Ygbr_W116BLBwnD4VNL/view?usp=drive_link",
        name="003_ATD_SRx4_finetune.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, ATD)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
        tolerance=3,
    )
