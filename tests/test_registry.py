from spandrel import ArchRegistry, ArchSupport

from .util import expect_error


def throw(*args):
    raise ValueError


def f(*args):
    return False


def mock_registry():
    r = ArchRegistry()
    r.add(
        ArchSupport(id="a", detect=f, load=throw),
        ArchSupport(id="b", detect=f, load=throw),
        ArchSupport(id="c", detect=f, load=throw),
        ArchSupport(id="d", detect=f, load=throw),
        ArchSupport(id="e", detect=f, load=throw),
        ArchSupport(id="f", detect=f, load=throw),
    )
    return r


def test_registry_order(snapshot):
    r = mock_registry()
    r.add(ArchSupport(id="test", detect=f, load=throw, before=("d",)))

    assert [a.id for a in r.architectures(order="insertion")] == snapshot
    assert [a.id for a in r.architectures(order="detection")] == snapshot


def test_registry_add_invalid(snapshot):
    r = mock_registry()
    original = r.architectures()

    with expect_error(snapshot):
        # Duplicate ID
        r.add(ArchSupport(id="b", detect=f, load=throw))

    assert original == r.architectures()

    with expect_error(snapshot):
        # Duplicate ID
        r.add(
            ArchSupport(id="test", detect=f, load=throw),
            ArchSupport(id="test", detect=f, load=throw),
        )

    assert original == r.architectures()

    with expect_error(snapshot):
        # Circular dependency
        r.add(
            ArchSupport(id="1", detect=f, load=throw, before=("2",)),
            ArchSupport(id="2", detect=f, load=throw, before=("1",)),
        )

    assert original == r.architectures()
