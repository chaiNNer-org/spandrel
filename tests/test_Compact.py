from spandrel.architectures.Compact import CompactArch, SRVGGNetCompact

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
        CompactArch(),
        lambda: SRVGGNetCompact(),
        lambda: SRVGGNetCompact(num_in_ch=1, num_out_ch=1),
        lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3),
        lambda: SRVGGNetCompact(num_in_ch=4, num_out_ch=4),
        lambda: SRVGGNetCompact(num_in_ch=1, num_out_ch=3),
        lambda: SRVGGNetCompact(num_feat=32),
        lambda: SRVGGNetCompact(num_conv=5),
        lambda: SRVGGNetCompact(upscale=3),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-AniScale.pth"
    )
    assert_size_requirements(file.load_model())


def test_Compact_realesr_general_x4v3(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SRVGGNetCompact)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_Compact_community(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-AniScale.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SRVGGNetCompact)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
