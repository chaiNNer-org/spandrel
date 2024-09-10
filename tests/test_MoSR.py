from spandrel.architectures.MoSR import MoSR, MoSRArch

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
        MoSRArch(),
        lambda: MoSR(),
        lambda: MoSR(in_ch=1, out_ch=1),
        lambda: MoSR(n_block=5),
        lambda: MoSR(dim=48),
        lambda: MoSR(upscale=2),
        lambda: MoSR(upsampler="dys"),
        lambda: MoSR(upsampler="gps"),
        lambda: MoSR(kernel_size=7),
        lambda: MoSR(expansion_ratio=2.0),
        lambda: MoSR(conv_ratio=2.0),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1zlpBQu74sguLCjvPpAL4p9t8QqcRktuL/view?usp=drive_link",
        name="4x_nomos2_mosr.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1_C9f6yHS-XZu0bSz3x9kvHy0M7gWHUWB/view?usp=drive_link",
        name="4x_nomos2_mosr_t.pth",
    )
    assert_size_requirements(file.load_model())


def test_2x_nomos2_mosr(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1MkVu7lIAyrGc1Rb7ediDKoNDcBc6PAdF/view?usp=drive_link",
        name="2x_nomos2_mosr.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MoSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_4x_nomos2_mosr(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1zlpBQu74sguLCjvPpAL4p9t8QqcRktuL/view?usp=drive_link",
        name="4x_nomos2_mosr.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MoSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
