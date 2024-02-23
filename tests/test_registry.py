from typing_extensions import override

from spandrel import Architecture, ArchRegistry, ArchSupport
from spandrel.__helpers.model_descriptor import ArchId

from .util import expect_error


class TestArch(Architecture):
    def __init__(self, id: str) -> None:
        super().__init__(
            id=id,
            detect=lambda _: False,
        )

    @override
    def load(self, state_dict):
        raise ValueError


def mock_registry():
    r = ArchRegistry()
    r.add(
        ArchSupport.from_architecture(TestArch(id="a")),
        ArchSupport.from_architecture(TestArch(id="b")),
        ArchSupport.from_architecture(TestArch(id="c")),
        ArchSupport.from_architecture(TestArch(id="d")),
        ArchSupport.from_architecture(TestArch(id="e")),
        ArchSupport.from_architecture(TestArch(id="f")),
    )
    return r


def test_registry_order(snapshot):
    r = mock_registry()
    r.add(ArchSupport.from_architecture(TestArch(id="test"), before=(ArchId("d"),)))

    assert [a.architecture.id for a in r.architectures(order="insertion")] == snapshot
    assert [a.architecture.id for a in r.architectures(order="detection")] == snapshot


def test_registry_add_invalid(snapshot):
    r = mock_registry()
    original = r.architectures()

    with expect_error(snapshot):
        # Duplicate ID
        r.add(ArchSupport.from_architecture(TestArch(id="b")))

    assert original == r.architectures()

    with expect_error(snapshot):
        # Duplicate ID
        r.add(
            ArchSupport.from_architecture(TestArch(id="test")),
            ArchSupport.from_architecture(TestArch(id="test")),
        )

    assert original == r.architectures()

    with expect_error(snapshot):
        # Circular dependency
        r.add(
            ArchSupport.from_architecture(TestArch(id="1"), before=(ArchId("2"),)),
            ArchSupport.from_architecture(TestArch(id="2"), before=(ArchId("1"),)),
        )

    assert original == r.architectures()
