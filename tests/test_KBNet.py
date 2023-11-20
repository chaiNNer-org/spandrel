from spandrel.architectures.KBNet import KBNet_l, KBNet_s, load

from .util import assert_loads_correctly


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
        lambda: KBNet_s(width=32, lightweight=True),
    )
