from spandrel.architectures.FLAVR import FLAVR, FLAVRArch

from .util import assert_loads_correctly, skip_if_unchanged

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        FLAVRArch(),
        lambda: FLAVR(),
        lambda: FLAVR(n_inputs=4, n_outputs=3),
        lambda: FLAVR(n_inputs=2, n_outputs=7),
        lambda: FLAVR(block="unet_34"),
        lambda: FLAVR(batchnorm=True),
        lambda: FLAVR(upmode="transpose", joinType="concat"),
        lambda: FLAVR(upmode="transpose", joinType="add"),
        lambda: FLAVR(upmode="direct", joinType="concat"),
        lambda: FLAVR(upmode="direct", joinType="add"),
    )
