from spandrel.architectures.LUDVAE import LUDVAE, LUDVAEArch

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
        LUDVAEArch(),
        lambda: LUDVAE(),
        lambda: LUDVAE(in_channel=4, filters_num=64),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/zhengdharia/LUD-VAE/raw/main/LUD_VAE_sidd/trained_models/LUDVAE_models/ludvae.pth",
        name="LUD_VAE_sidd.pth",
    )
    assert_size_requirements(file.load_model())


def test_LUD_VAE_sidd(snapshot):
    file = ModelFile.from_url(
        "https://github.com/zhengdharia/LUD-VAE/raw/main/LUD_VAE_sidd/trained_models/LUDVAE_models/ludvae.pth",
        name="LUD_VAE_sidd.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, LUDVAE)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_64],
    )
