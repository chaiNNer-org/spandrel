from spandrel import ModelLoader
from spandrel.architectures.Compact import SRVGGNetCompact

from .util import ImageTestNames, ModelFile, compare_images_to_results, disallowed_props


def test_Compact_realesr_general_x4v3(snapshot):
    file = ModelFile.from_url(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SRVGGNetCompact)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )


def test_Compact_community(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-AniScale.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, SRVGGNetCompact)
    assert compare_images_to_results(
        file.name,
        model.model,
        [ImageTestNames.SR_16, ImageTestNames.SR_32, ImageTestNames.SR_64],
    )
