from spandrel.architectures.DAT import DAT, load

from .util import assert_loads_correctly


def test_DAT_load():
    assert_loads_correctly(
        load,
        lambda: DAT(),
        lambda: DAT(embed_dim=60),
        lambda: DAT(in_chans=1),
        lambda: DAT(in_chans=4),
        lambda: DAT(depth=[2, 3], num_heads=[2, 5]),
        lambda: DAT(depth=[2, 3, 4, 2], num_heads=[2, 3, 2, 2]),
        lambda: DAT(depth=[2, 3, 4, 2, 5], num_heads=[2, 3, 2, 2, 3]),
        lambda: DAT(upsampler="pixelshuffle", upscale=1),
        lambda: DAT(upsampler="pixelshuffle", upscale=2),
        lambda: DAT(upsampler="pixelshuffle", upscale=3),
        lambda: DAT(upsampler="pixelshuffle", upscale=4),
        lambda: DAT(upsampler="pixelshuffle", upscale=8),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=1),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=2),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=3),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=4),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=8),
        lambda: DAT(resi_connection="3conv"),
        lambda: DAT(qkv_bias=False),
        condition=lambda a, b: (
            a.num_layers == b.num_layers
            and a.upscale == b.upscale
            and a.upsampler == b.upsampler
            and a.embed_dim == b.embed_dim
            and a.num_features == b.num_features
        ),
    )
