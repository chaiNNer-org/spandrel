from spandrel.architectures.SPSR import SPSR, SPSRArch

from .util import assert_loads_correctly


def test_SPSR_load():
    assert_loads_correctly(
        SPSRArch(),
        lambda: SPSR(
            in_nc=3,
            out_nc=3,
            num_filters=64,
            num_blocks=16,
        ),
        lambda: SPSR(
            in_nc=1,
            out_nc=3,
            num_filters=32,
            num_blocks=8,
        ),
        lambda: SPSR(
            in_nc=4,
            out_nc=4,
            num_filters=32,
            num_blocks=8,
        ),
        # upscale
        lambda: SPSR(
            in_nc=3,
            out_nc=3,
            num_filters=64,
            num_blocks=16,
            upscale=1,
            upsample_mode="upconv",
        ),
        lambda: SPSR(
            in_nc=3,
            out_nc=3,
            num_filters=64,
            num_blocks=16,
            upscale=2,
            upsample_mode="upconv",
        ),
        lambda: SPSR(
            in_nc=3,
            out_nc=3,
            num_filters=64,
            num_blocks=16,
            upscale=4,
            upsample_mode="upconv",
        ),
        lambda: SPSR(
            in_nc=3,
            out_nc=3,
            num_filters=64,
            num_blocks=16,
            upscale=8,
            upsample_mode="upconv",
        ),
        lambda: SPSR(
            in_nc=3,
            out_nc=3,
            num_filters=64,
            num_blocks=16,
            upscale=1,
            upsample_mode="pixelshuffle",
        ),
        lambda: SPSR(
            in_nc=3,
            out_nc=3,
            num_filters=64,
            num_blocks=16,
            upscale=2,
            upsample_mode="pixelshuffle",
        ),
        lambda: SPSR(
            in_nc=3,
            out_nc=3,
            num_filters=64,
            num_blocks=16,
            upscale=4,
            upsample_mode="pixelshuffle",
        ),
        lambda: SPSR(
            in_nc=3,
            out_nc=3,
            num_filters=64,
            num_blocks=16,
            upscale=8,
            upsample_mode="pixelshuffle",
        ),
    )
