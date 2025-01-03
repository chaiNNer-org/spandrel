from spandrel.architectures.MoESR import MoESR, MoESRArch

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
        MoESRArch(),
        lambda: MoESR(),
        lambda: MoESR(in_ch=1, out_ch=1),
        lambda: MoESR(n_blocks=10),
        lambda: MoESR(n_block=5),
        lambda: MoESR(dim=32),
        lambda: MoESR(scale=2),
        lambda: MoESR(scale=1, upsampler="conv"),
        lambda: MoESR(upsampler="pixelshuffle"),
        lambda: MoESR(upsampler="dysample"),
        lambda: MoESR(expansion_factor=2.0),
        lambda: MoESR(expansion_msg=1.0),
        lambda: MoESR(upsample_dim=32),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/the-database/traiNNer-redux/releases/download/pretrained-models/4x_DF2K_MoESR_500k.safetensors",
        name="4x_DF2K_MoESR_500k.safetensors",
    )
    assert_size_requirements(file.load_model())


def test_4x_df2k_moesr(snapshot):
    file = ModelFile.from_url(
        "https://github.com/the-database/traiNNer-redux/releases/download/pretrained-models/4x_DF2K_MoESR_500k.safetensors",
        name="4x_DF2K_MoESR_500k.safetensors",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MoESR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
