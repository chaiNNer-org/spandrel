from spandrel_extra_arches.architectures.AdaCode import AdaCode, AdaCodeArch
from tests.test_GFPGAN import disallowed_props

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
)


def test_load():
    assert_loads_correctly(
        AdaCodeArch(),
        lambda: AdaCode(),
        lambda: AdaCode(in_channel=1),
        lambda: AdaCode(in_channel=4),
        lambda: AdaCode(gt_resolution=512),
        lambda: AdaCode(gt_resolution=128),
        lambda: AdaCode(LQ_stage=True, scale_factor=2),
        lambda: AdaCode(LQ_stage=True, scale_factor=4),
        lambda: AdaCode(LQ_stage=True, scale_factor=8),
        lambda: AdaCode(norm_type="gn"),
        lambda: AdaCode(norm_type="bn"),
        lambda: AdaCode(norm_type="in"),
        # lambda: AdaCode(weight_softmax=True),
        lambda: AdaCode(codebook_params=[[32, 1024, 512]]),
        lambda: AdaCode(codebook_params=[[32, 512, 256]]),
        lambda: AdaCode(codebook_params=[[64, 512, 256], [32, 1024, 512]]),
        ignore_parameters={
            # there are multiple equivalent codebook_params for some models
            "codebook_params"
        },
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/kechunl/AdaCode/releases/download/v0-pretrain_models/AdaCode_SR_X2_model_g.pth"
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/kechunl/AdaCode/releases/download/v0-pretrain_models/AdaCode_SR_X4_model_g.pth"
    )
    assert_size_requirements(file.load_model())


def test_AdaCode_SR_X2_model_g(snapshot):
    file = ModelFile.from_url(
        "https://github.com/kechunl/AdaCode/releases/download/v0-pretrain_models/AdaCode_SR_X2_model_g.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, AdaCode)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_32, TestImage.SR_64],
    )


def test_AdaCode_SR_X4_model_g(snapshot):
    file = ModelFile.from_url(
        "https://github.com/kechunl/AdaCode/releases/download/v0-pretrain_models/AdaCode_SR_X4_model_g.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, AdaCode)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
