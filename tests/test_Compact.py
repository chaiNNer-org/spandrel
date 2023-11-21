from spandrel import ModelLoader
from spandrel.architectures.Compact import SRVGGNetCompact, load

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    disallowed_props,
)


def test_Compact_load():
    assert_loads_correctly(
        load,
        lambda: SRVGGNetCompact(),
        lambda: SRVGGNetCompact(num_in_ch=1, num_out_ch=1),
        lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3),
        lambda: SRVGGNetCompact(num_in_ch=4, num_out_ch=4),
        lambda: SRVGGNetCompact(num_in_ch=1, num_out_ch=3),
        lambda: SRVGGNetCompact(num_feat=32),
        lambda: SRVGGNetCompact(num_conv=5),
        lambda: SRVGGNetCompact(upscale=3),
        condition=lambda a, b: (
            a.upscale == b.upscale
            and a.num_in_ch == b.num_in_ch
            and a.num_out_ch == b.num_out_ch
            and a.num_feat == b.num_feat
            and a.num_conv == b.num_conv
        ),
    )


def test_Compact_realesr_general_x4v3(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    )
    model = ModelLoader().load_from_file(file.path)
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
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SRVGGNetCompact)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )
