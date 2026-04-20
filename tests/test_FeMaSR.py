from spandrel_extra_arches.architectures.FeMaSR import FeMaSR, FeMaSRArch

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
        FeMaSRArch(),
        lambda: FeMaSR(),
        lambda: FeMaSR(in_channel=1),
        lambda: FeMaSR(in_channel=4),
        lambda: FeMaSR(gt_resolution=512),
        lambda: FeMaSR(gt_resolution=128),
        lambda: FeMaSR(LQ_stage=True, scale_factor=2),
        lambda: FeMaSR(LQ_stage=True, scale_factor=4),
        lambda: FeMaSR(LQ_stage=True, scale_factor=8, codebook_params=[[8, 32, 32]]),
        lambda: FeMaSR(norm_type="gn"),
        lambda: FeMaSR(norm_type="bn"),
        lambda: FeMaSR(norm_type="in"),
        lambda: FeMaSR(codebook_params=[[32, 1024, 512]]),
        lambda: FeMaSR(codebook_params=[[32, 512, 256]]),
        lambda: FeMaSR(codebook_params=[[64, 512, 256], [32, 1024, 512]]),
        ignore_parameters={
            # there are multiple equivalent codebook_params for some models
            "codebook_params"
        },
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_HRP_model_g.pth"
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth"
    )
    assert_size_requirements(file.load_model())


def test_train():
    assert_training(FeMaSRArch(), FeMaSR())


def test_FeMaSR_1x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_HRP_model_g.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, FeMaSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_8, TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_FeMaSR_2x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, FeMaSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_32, TestImage.SR_64],
    )


def test_FeMaSR_4x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, FeMaSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
