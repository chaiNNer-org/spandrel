from spandrel import ModelLoader
from spandrel.architectures.LaMa import LaMa

from .util import ModelFile, disallowed_props


def test_LaMa(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, LaMa)
