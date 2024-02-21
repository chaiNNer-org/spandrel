from spandrel.architectures.NAFNet import NAFNet, NAFNetArch

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
        NAFNetArch(),
        lambda: NAFNet(
            img_channel=3,
            width=32,
            middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2],
        ),
        lambda: NAFNet(
            img_channel=3,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 28],
            dec_blk_nums=[1, 1, 1, 1],
        ),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view",
        name="NAFNet-GoPro-width32.pth",
    )
    assert_size_requirements(file.load_model())


def test_NAFNet_GoPro_width32(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view",
        name="NAFNet-GoPro-width32.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, NAFNet)
    assert_image_inference(
        file,
        model,
        [TestImage.BLURRY_FACE, TestImage.SR_32],
    )
