from spandrel.architectures.OmniSR import OmniSR, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
)


def test_OmniSR_load():
    assert_loads_correctly(
        load,
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
        condition=lambda a, b: (
            a.res_num == b.res_num
            and a.up_scale == b.up_scale
            and a.window_size == b.window_size
        ),
    )


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
