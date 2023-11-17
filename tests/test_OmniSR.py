from spandrel import ModelLoader
from spandrel.architectures.OmniSR import OmniSR

from .util import ModelFile, disallowed_props


def test_OmniSR_community1(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Phhofm/models/raw/main/2xHFA2kAVCOmniSR/2xHFA2kAVCOmniSR.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, OmniSR)


def test_OmniSR_community2(snapshot):
    file = ModelFile.from_url(
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-ardo.pth"
    )
    model = ModelLoader().load_from_file(file.path)
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, OmniSR)
