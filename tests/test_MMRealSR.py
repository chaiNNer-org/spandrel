from spandrel.architectures.MMRealSR import MMRealSR, MMRealSRArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
)


def test_MMRealSR_load():
    assert_loads_correctly(
        MMRealSRArch(),
        # num_block=2 is used everywhere to make tests faster
        lambda: MMRealSR(num_block=2, num_in_ch=3, num_out_ch=3),
        lambda: MMRealSR(num_block=2, num_in_ch=3, num_out_ch=3, scale=1),
        lambda: MMRealSR(num_block=2, num_in_ch=3, num_out_ch=3, scale=2),
        lambda: MMRealSR(num_block=2, num_in_ch=3, num_out_ch=3, scale=4),
        lambda: MMRealSR(num_block=2, num_in_ch=4, num_out_ch=4, scale=1),
        lambda: MMRealSR(num_block=2, num_in_ch=1, num_out_ch=1, scale=4),
        lambda: MMRealSR(num_block=2, num_in_ch=3, num_out_ch=3, num_feat=16),
        lambda: MMRealSR(num_block=2, num_in_ch=3, num_out_ch=3, num_grow_ch=4),
        lambda: MMRealSR(
            num_block=2,
            num_in_ch=3,
            num_out_ch=3,
            num_feats=[32, 64, 128],
            num_blocks=[1, 3, 2],
            downscales=[2, 2, 2],
        ),
        # official configs
        # MMRealSRNet_x4
        lambda: MMRealSR(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            de_net_type="DEResNet",
            num_degradation=2,
            degradation_degree_actv="sigmoid",
            num_feats=[64, 64, 64, 128],
            num_blocks=[2, 2, 2, 2],
            downscales=[1, 1, 2, 1],
        ),
        # MMRealSRGAN_x4
        lambda: MMRealSR(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            de_net_type="DEResNet",
            num_degradation=2,
            degradation_degree_actv="sigmoid",
            num_feats=[64, 64, 64, 128],
            num_blocks=[2, 2, 2, 2],
            downscales=[1, 1, 2, 1],
        ),
        condition=lambda a, b: (
            a.scale == b.scale
            and a.num_degradation == b.num_degradation
            and a.num_block == b.num_block
        ),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/TencentARC/MM-RealSR/releases/download/v1.0.0/MMRealSRGAN.pth"
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/TencentARC/MM-RealSR/releases/download/v1.0.0/MMRealSRNet.pth"
    )
    assert_size_requirements(file.load_model())


def test_MMRealSRGAN(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TencentARC/MM-RealSR/releases/download/v1.0.0/MMRealSRGAN.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MMRealSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_MMRealSRGAN_ModulationBest(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TencentARC/MM-RealSR/releases/download/v1.0.0/MMRealSRGAN_ModulationBest.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MMRealSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_MMRealSRNet(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TencentARC/MM-RealSR/releases/download/v1.0.0/MMRealSRNet.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MMRealSR)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
