from spandrel.architectures.KBNet import KBNet_l, KBNet_s, KBNetArch

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
        KBNetArch(),
        lambda: KBNet_l(),
        lambda: KBNet_l(inp_channels=4, out_channels=1),
        lambda: KBNet_l(inp_channels=1, out_channels=4),
        lambda: KBNet_l(dim=24),
        lambda: KBNet_l(
            num_blocks=[1, 2, 3, 4],
            num_refinement_blocks=10,
            heads=[1, 2, 3, 4],
        ),
        lambda: KBNet_l(ffn_expansion_factor=2, bias=True),
        # this reduces RAM, so we use it everywhere
        lambda: KBNet_s(width=32),
        lambda: KBNet_s(width=32, img_channel=1),
        lambda: KBNet_s(width=32, img_channel=4),
        lambda: KBNet_s(width=32, middle_blk_num=3),
        lambda: KBNet_s(width=32, enc_blk_nums=[1, 2, 3, 4], dec_blk_nums=[1, 2, 3, 4]),
        lambda: KBNet_s(width=32, ffn_scale=3),
        lambda: KBNet_s(width=32, lightweight=True),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/kbnet/1x-KBNet_derain.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/kbnet/1x-KBNet_sidd.pth",
    )
    assert_size_requirements(file.load_model())


def test_train():
    assert_training(KBNetArch(), KBNet_s())
    assert_training(KBNetArch(), KBNet_l())


def test_derain(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/kbnet/1x-KBNet_derain.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, KBNet_l)
    assert_image_inference(file, model, [TestImage.SR_32])


def test_sidd(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/kbnet/1x-KBNet_sidd.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, KBNet_s)
    assert_image_inference(file, model, [TestImage.JPEG_15])
