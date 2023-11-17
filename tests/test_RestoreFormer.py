from spandrel import ModelLoader
from spandrel.architectures.RestoreFormer import RestoreFormer

from .util import ModelFile, disallowed_props


def test_RestoreFormer(snapshot):
    file = ModelFile.from_url(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RestoreFormer)
