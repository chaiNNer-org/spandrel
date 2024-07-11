from spandrel.architectures.LaMa import LaMa, LaMaArch

from .util import (
    ModelFile,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        LaMaArch(),
        lambda: LaMa(),
        lambda: LaMa(in_nc=4),
        lambda: LaMa(out_nc=1),
        lambda: LaMa(in_nc=1, out_nc=1),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
    )
    assert_size_requirements(file.load_model())


def test_LaMa(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, LaMa)
