__version__ = "0.0.4"

from .__helpers.canonicalize import canonicalize_state_dict
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
    "canonicalize_state_dict",
    "FaceSRModelDescriptor",
    "InpaintModelDescriptor",
    "MAIN_REGISTRY",
    "ModelBase",
    "ModelDescriptor",
    "ModelLoader",
    "RestorationModelDescriptor",
    "SizeRequirements",
    "SRModelDescriptor",
    "StateDict",
    "UnsupportedModelError",
]
