from spandrel import ModelLoader
from spandrel.architectures.Swin2SR import Swin2SR

from .util import ModelFile, compare_images_to_results, disallowed_props


def test_Swin2SR_4x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X4_64.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, Swin2SR)
    assert compare_images_to_results(file.name, model.model)
