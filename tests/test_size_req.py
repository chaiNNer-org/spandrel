import pytest

from spandrel import SizeRequirements


def test_size_req_init():
    a = SizeRequirements(minimum=1, multiple_of=2)
    assert a.minimum == 2
    assert a.multiple_of == 2

    with pytest.raises(AssertionError):
        SizeRequirements(minimum=-1)
    with pytest.raises(AssertionError):
        SizeRequirements(multiple_of=0)
    with pytest.raises(AssertionError):
        SizeRequirements(multiple_of=-1)
