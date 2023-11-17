from spandrel import ModelLoader
from spandrel.architectures.CodeFormer import CodeFormer

from .util import ModelFile, disallowed_props


def test_CodeFormer(snapshot):
    file = ModelFile.from_url(
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, CodeFormer)
