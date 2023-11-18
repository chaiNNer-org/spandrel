from spandrel import ModelLoader
from spandrel.architectures.KBCNN import KBNet_l, KBNet_s, load

from .util import ModelFile, assert_loads_correctly, disallowed_props


def test_KBCNN_load():
    assert_loads_correctly(
        load,
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
    )


def test_KBCNN_deblur(snapshot):
    file = ModelFile.from_url("https://TODO.com/kbcnn_defocus.pth")
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, KBNet_l)
    # assert_image_inference(file, model, [TestImage.JPEG_15])


def test_KBCNN_derain(snapshot):
    file = ModelFile.from_url("https://TODO.com/kbcnn_derain.pth")
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    # assert isinstance(model.model, KBNet_l)
