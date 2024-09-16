"""
Spandrel extra arches contains more architectures for `spandrel`.

All architectures in this library are registered in the `EXTRA_REGISTRY` dictionary.
"""

from .__helper import EXTRA_REGISTRY, install

__version__ = "0.2.0"

__all__ = [
    "EXTRA_REGISTRY",
    "install",
]
