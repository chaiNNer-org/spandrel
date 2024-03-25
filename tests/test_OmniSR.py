from spandrel.architectures.OmniSR import OmniSR, OmniSRArch

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
        OmniSRArch(),
        lambda: OmniSR(),
        lambda: OmniSR(num_in_ch=1, num_out_ch=1),
        lambda: OmniSR(num_in_ch=3, num_out_ch=3),
        lambda: OmniSR(num_in_ch=4, num_out_ch=4),
        lambda: OmniSR(num_in_ch=1, num_out_ch=3),
        lambda: OmniSR(num_feat=32),
        lambda: OmniSR(block_num=2),
        lambda: OmniSR(pe=False),
        lambda: OmniSR(bias=False),
        lambda: OmniSR(window_size=5),
        lambda: OmniSR(res_num=3),
        lambda: OmniSR(up_scale=5),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/omnisr/4x-OmniSR-DF2K.pth"
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/Phhofm/models/raw/main/2xHFA2kAVCOmniSR/2xHFA2kAVCOmniSR.pth"
    )
    assert_size_requirements(file.load_model())


def test_OmniSR_official_x4(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/omnisr/4x-OmniSR-DF2K.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, OmniSR)


def test_OmniSR_official_x3(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/omnisr/3x-OmniSR-DIV2K.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, OmniSR)


def test_OmniSR_official_x2(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/omnisr/2x-OmniSR-DIV2K.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, OmniSR)


def test_OmniSR_community1(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Phhofm/models/raw/main/2xHFA2kAVCOmniSR/2xHFA2kAVCOmniSR.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, OmniSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_OmniSR_community2(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-ardo.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, OmniSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
