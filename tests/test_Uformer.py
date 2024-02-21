from spandrel.architectures.Uformer import Uformer, UformerArch

from .util import assert_loads_correctly


def test_load():
    assert_loads_correctly(
        UformerArch(),
        lambda: Uformer(),
        lambda: Uformer(in_chans=1),
        lambda: Uformer(dd_in=1),
        # embed_dim=4 makes tests way faster
        lambda: Uformer(embed_dim=4),
        lambda: Uformer(embed_dim=4, depths=[2, 2, 3, 2, 2, 1, 2, 3, 2]),
        lambda: Uformer(embed_dim=4, num_heads=[2, 3, 1, 5, 7, 7, 6, 3, 1]),
        lambda: Uformer(embed_dim=4, win_size=9),
        lambda: Uformer(embed_dim=4, mlp_ratio=5),
        lambda: Uformer(embed_dim=4, qkv_bias=False, token_mlp="leff"),
        lambda: Uformer(embed_dim=4, qkv_bias=False, token_mlp="fastleff"),
        lambda: Uformer(embed_dim=4, qkv_bias=False, token_mlp="mlp"),
        lambda: Uformer(embed_dim=4, token_mlp="leff"),
        lambda: Uformer(embed_dim=4, token_mlp="fastleff"),
        lambda: Uformer(embed_dim=4, token_mlp="mlp"),
        lambda: Uformer(embed_dim=4, token_projection="linear"),
        lambda: Uformer(embed_dim=4, token_projection="conv"),
        lambda: Uformer(embed_dim=4, modulator=True),
        lambda: Uformer(embed_dim=4, cross_modulator=True),
    )
