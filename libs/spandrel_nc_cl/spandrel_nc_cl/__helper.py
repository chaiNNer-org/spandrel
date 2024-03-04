from spandrel import ArchRegistry, ArchSupport
from spandrel_nc import NC_REGISTRY

from .architectures import (
    FeMaSR,
    M3SNet,
    Restormer,
)

NC_CL_REGISTRY = ArchRegistry()

NC_CL_REGISTRY.add(
    *NC_REGISTRY,
    ArchSupport.from_architecture(FeMaSR.FeMaSRArch()),
    ArchSupport.from_architecture(M3SNet.M3SNetArch()),
    ArchSupport.from_architecture(Restormer.RestormerArch()),
)
