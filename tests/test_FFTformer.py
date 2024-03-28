from spandrel.architectures.FFTformer import FFTformer, FFTformerArch

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
        FFTformerArch(),
        lambda: FFTformer(),
        lambda: FFTformer(dim=64, inp_channels=4, out_channels=1),
        lambda: FFTformer(num_blocks=[3, 5, 7], ffn_expansion_factor=2),
        lambda: FFTformer(num_refinement_blocks=2),
        lambda: FFTformer(bias=True),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/kkkls/FFTformer/releases/download/pretrain_model/fftformer_GoPro.pth",
        name="fftformer_GoPro.pth",
    )
    assert_size_requirements(file.load_model())


def test_fftformer_GoPro(snapshot):
    file = ModelFile.from_url(
        "https://github.com/kkkls/FFTformer/releases/download/pretrain_model/fftformer_GoPro.pth",
        name="fftformer_GoPro.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, FFTformer)
    assert_image_inference(file, model, [TestImage.SR_32])
