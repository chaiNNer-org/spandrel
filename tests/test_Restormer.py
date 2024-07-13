from spandrel_extra_arches.architectures.Restormer import Restormer, RestormerArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    assert_training,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        RestormerArch(),
        lambda: Restormer(),
        lambda: Restormer(dim=64, inp_channels=4, out_channels=1),
        lambda: Restormer(num_refinement_blocks=2),
        lambda: Restormer(ffn_expansion_factor=2, bias=True),
        lambda: Restormer(LayerNorm_type="WithBias", dual_pixel_task=True),
        lambda: Restormer(LayerNorm_type="BiasFree", dual_pixel_task=False),
        lambda: Restormer(num_blocks=[2, 3, 5, 7], heads=[11, 13, 17, 19]),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/swz30/Restormer/releases/download/v1.0/deraining.pth",
        name="restormer_deraining.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/swz30/Restormer/releases/download/v1.0/dual_pixel_defocus_deblurring.pth",
        name="restormer_dual_pixel_defocus_deblurring.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth",
        name="restormer_motion_deblurring.pth",
    )
    assert_size_requirements(file.load_model())


def test_train():
    assert_training(RestormerArch(), Restormer())


def test_deraining(snapshot):
    file = ModelFile.from_url(
        "https://github.com/swz30/Restormer/releases/download/v1.0/deraining.pth",
        name="restormer_deraining.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Restormer)


def test_dual_pixel_defocus_deblurring(snapshot):
    file = ModelFile.from_url(
        "https://github.com/swz30/Restormer/releases/download/v1.0/dual_pixel_defocus_deblurring.pth",
        name="restormer_dual_pixel_defocus_deblurring.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Restormer)


def test_motion_deblurring(snapshot):
    file = ModelFile.from_url(
        "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth",
        name="restormer_motion_deblurring.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Restormer)


def test_gaussian_color_denoising_blind(snapshot):
    file = ModelFile.from_url(
        "https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth",
        name="restormer_gaussian_color_denoising_blind.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Restormer)
    assert_image_inference(file, model, [TestImage.JPEG_15])


def test_gaussian_gray_denoising_sigma25(snapshot):
    file = ModelFile.from_url(
        "https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_gray_denoising_sigma25.pth",
        name="restormer_gaussian_gray_denoising_sigma25.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Restormer)
