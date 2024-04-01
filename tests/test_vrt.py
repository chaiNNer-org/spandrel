from spandrel_extra_arches.architectures.VRT import VRT, VRTArch

from .util import assert_loads_correctly, skip_if_unchanged

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        VRTArch(),
        lambda: VRT(),
        lambda: VRT(pa_frames=0),
        lambda: VRT(pa_frames=6),
        lambda: VRT(deformable_groups=8),
        lambda: VRT(nonblind_denoising=True),
        lambda: VRT(in_chans=1),
        lambda: VRT(out_chans=1),
        lambda: VRT(qkv_bias=False),
        lambda: VRT(upscale=1),
        lambda: VRT(upscale=2),
        lambda: VRT(upscale=3),
        lambda: VRT(upscale=4),
        lambda: VRT(upscale=8),
        lambda: VRT(mlp_ratio=1.5),
    )
