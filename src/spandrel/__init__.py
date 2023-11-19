__version__ = "0.0.2"

from .__helpers.loader import ModelLoader
from .__helpers.main_registry import MAIN_REGISTRY
from .__helpers.model_descriptor import (
    FaceSRModelDescriptor,
    InpaintModelDescriptor,
    ModelBase,
    ModelDescriptor,
    RestorationModelDescriptor,
    SizeRequirements,
    SRModelDescriptor,
    StateDict,
)
from .__helpers.registry import ArchRegistry, ArchSupport, UnsupportedModelError

__all__ = [
    "ArchRegistry",
    "ArchSupport",
    "RestorationModelDescriptor",
    "FaceSRModelDescriptor",
    "InpaintModelDescriptor",
    "MAIN_REGISTRY",
    "ModelBase",
    "ModelDescriptor",
    "ModelLoader",
    "SizeRequirements",
    "SRModelDescriptor",
    "StateDict",
    "UnsupportedModelError",
]
