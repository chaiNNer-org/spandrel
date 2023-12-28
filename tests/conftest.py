import pytest


@pytest.fixture(scope="function", autouse=True)
def auto_seed_rngs():
    """
    Ensure tests are deterministic by seeding RNGs.

    This is automagically called for each test by py.test.
    """
    from tests.util import seed_rngs

    seed_rngs(2812)
