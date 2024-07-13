from spandrel_extra_arches.architectures.MAT import MAT, MATArch

from .util import (
    ModelFile,
    assert_loads_correctly,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        MATArch(),
        lambda: MAT(),
        check_safe_tensors=False,
    )


def test_MAT(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, MAT)
