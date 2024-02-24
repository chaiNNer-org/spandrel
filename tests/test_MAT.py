from spandrel_nc.architectures.MAT import MAT

from .util import ModelFile, disallowed_props


def test_MAT(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MAT)
