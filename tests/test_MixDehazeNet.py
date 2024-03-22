from spandrel.architectures.MixDehazeNet import MixDehazeNet, MixDehazeNetArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
)


def test_load():
    assert_loads_correctly(
        MixDehazeNetArch(),
        lambda: MixDehazeNet(in_chans=4, out_chans=2),
        lambda: MixDehazeNet(embed_dims=[12, 60, 32, 60, 12], depths=[1, 2, 3, 4, 5]),
        # official configs
        lambda: MixDehazeNet(embed_dims=[24, 48, 96, 48, 24], depths=[1, 1, 2, 1, 1]),
        lambda: MixDehazeNet(embed_dims=[24, 48, 96, 48, 24], depths=[2, 2, 4, 2, 2]),
        lambda: MixDehazeNet(embed_dims=[24, 48, 96, 48, 24], depths=[4, 4, 8, 4, 4]),
        lambda: MixDehazeNet(embed_dims=[24, 48, 96, 48, 24], depths=[8, 8, 16, 8, 8]),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1-SH2I3pkLjWouKcld5JnMdSNLPJUwCP8/view?usp=drive_link",
        name="MixDehazeNet-s indoor.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1Ds-uyUQg2VBWga6ZOXKVUeefhR-QTu-O/view?usp=drive_link",
        name="MixDehazeNet-b indoor.pth",
    )
    assert_size_requirements(file.load_model())


def test_b_indoor(snapshot):
    file = ModelFile.from_url(
        "https://drive.google.com/file/d/1Ds-uyUQg2VBWga6ZOXKVUeefhR-QTu-O/view?usp=drive_link",
        name="MixDehazeNet-b indoor.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MixDehazeNet)
    assert_image_inference(file, model, [TestImage.HAZE])
