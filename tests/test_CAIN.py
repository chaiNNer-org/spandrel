from spandrel.architectures.CAIN import CAIN, CAIN_EncDec, CAIN_NoCA, CAINArch

from .util import ModelFile, assert_loads_correctly, disallowed_props, skip_if_unchanged

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        CAINArch(),
        lambda: CAIN(),
        lambda: CAIN(depth=2),
        lambda: CAIN_NoCA(),
        lambda: CAIN_NoCA(depth=2),
        lambda: CAIN_EncDec(),
        lambda: CAIN_EncDec(start_filters=16),
        lambda: CAIN_EncDec(up_mode="shuffle"),
        lambda: CAIN_EncDec(up_mode="transpose"),
        lambda: CAIN_EncDec(up_mode="direct"),
    )


def test_pretrain(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1Oqv1PrBHAAq23Lj1Z3noy9VNtCgPqoqg/view?usp=drive_link",
        name="pretrained_cain.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
