from spandrel.architectures.RetinexFormer import RetinexFormer, RetinexFormerArch

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
        RetinexFormerArch(),
        lambda: RetinexFormer(),
        lambda: RetinexFormer(stage=1),
        lambda: RetinexFormer(in_channels=1, out_channels=1, n_feat=20),
        lambda: RetinexFormer(num_blocks=[3, 5, 7]),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1oxvPPfhbOwZURTFenWnFp3H3Lakkqw3t/view?usp=drive_link",
        name="retinexFormer_FiveK.pth",
    )
    assert_size_requirements(file.load_model())


def test_train():
    assert_training(RetinexFormerArch(), RetinexFormer())


def test_FiveK(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1oxvPPfhbOwZURTFenWnFp3H3Lakkqw3t/view?usp=drive_link",
        name="retinexFormer_FiveK.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RetinexFormer)
    assert_image_inference(file, model, [TestImage.LOW_LIGHT_FIVE_K])


def test_NTIRE(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1K-QR-A_CPe6iAgjE6_04Q20DkkVwaVta/view?usp=drive_link",
        name="retinexFormer_NTIRE.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RetinexFormer)
    return
    assert_image_inference(file, model, [TestImage.JPEG_15])
