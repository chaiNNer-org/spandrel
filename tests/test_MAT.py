from spandrel import ModelLoader
from spandrel.architectures.MAT import MAT

from .util import ModelFile, disallowed_props


def test_MAT(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MAT)
