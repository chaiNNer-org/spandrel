from spandrel.architectures.DIS import DIS, DISArch

from .util import (
    assert_loads_correctly,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        DISArch(),
        # Defaults
        lambda: DIS(),
        # Different channel counts
        lambda: DIS(in_channels=1, out_channels=1),
        lambda: DIS(in_channels=4, out_channels=3),
        # Different feature counts
        lambda: DIS(num_features=16),
        lambda: DIS(num_features=64),
        # Different block counts
        lambda: DIS(num_blocks=2),
        lambda: DIS(num_blocks=8),
        lambda: DIS(num_blocks=12),
        # Different scales
        lambda: DIS(scale=1),
        lambda: DIS(scale=2),
        lambda: DIS(scale=3),
        lambda: DIS(scale=4),
        # Depthwise separable variant
        lambda: DIS(use_depthwise=True),
        lambda: DIS(use_depthwise=True, num_blocks=8),
        lambda: DIS(use_depthwise=True, scale=2),
        lambda: DIS(use_depthwise=True, scale=3),
        lambda: DIS(use_depthwise=True, scale=1),
    )
