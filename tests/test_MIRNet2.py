from spandrel_extra_arches.architectures.MIRNet2 import MIRNet2, MIRNet2Arch

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
        MIRNet2Arch(),
        lambda: MIRNet2(),
        lambda: MIRNet2(inp_channels=1, n_feat=32),
        lambda: MIRNet2(out_channels=1, bias=True),
        lambda: MIRNet2(n_MRB=3),
        lambda: MIRNet2(chan_factor=2),
        lambda: MIRNet2(
            # https://github.com/swz30/MIRNetv2/blob/main/Defocus_Deblurring/Options/DefocusDeblur_DualPixel_16bit_MIRNet_v2.yml
            inp_channels=6,
            out_channels=3,
            n_feat=80,
            chan_factor=1.5,
            n_MRB=2,
            task="defocus_deblurring",
        ),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/dual_pixel_defocus_deblurring.pth",
        name="MIRNet2_dual_pixel_defocus_deblurring.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/enhancement_lol.pth",
        name="MIRNet2_enhancement_lol.pth",
    )
    assert_size_requirements(file.load_model())


def test_train():
    assert_training(MIRNet2Arch(), MIRNet2())


def test_dual_pixel_defocus_deblurring(snapshot):
    file = ModelFile.from_url(
        "https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/dual_pixel_defocus_deblurring.pth",
        name="MIRNet2_dual_pixel_defocus_deblurring.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MIRNet2)


def test_enhancement_lol(snapshot):
    file = ModelFile.from_url(
        "https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/enhancement_lol.pth",
        name="MIRNet2_enhancement_lol.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MIRNet2)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_32],
    )
