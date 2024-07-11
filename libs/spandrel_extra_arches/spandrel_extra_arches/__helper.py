from __future__ import annotations

from spandrel import (
    MAIN_REGISTRY,
    ArchRegistry,
    ArchSupport,
)

from .architectures import (
    MAT,
    AdaCode,
    CodeFormer,
    DDColor,
    FeMaSR,
    M3SNet,
    MIRNet2,
    MPRNet,
    Restormer,
    SRFormer,
)

EXTRA_REGISTRY = ArchRegistry()
"""
The registry of all architectures in this library.

Use ``MAIN_REGISTRY.add(*EXTRA_REGISTRY)`` to add all architectures to the main registry of `spandrel`.
"""

EXTRA_REGISTRY.add(
    ArchSupport.from_architecture(SRFormer.SRFormerArch()),
    ArchSupport.from_architecture(CodeFormer.CodeFormerArch()),
    ArchSupport.from_architecture(MAT.MATArch()),
    ArchSupport.from_architecture(DDColor.DDColorArch()),
    ArchSupport.from_architecture(AdaCode.AdaCodeArch()),
    ArchSupport.from_architecture(FeMaSR.FeMaSRArch()),
    ArchSupport.from_architecture(M3SNet.M3SNetArch()),
    ArchSupport.from_architecture(Restormer.RestormerArch()),
    ArchSupport.from_architecture(MPRNet.MPRNetArch()),
    ArchSupport.from_architecture(MIRNet2.MIRNet2Arch()),
)


def install(*, ignore_duplicates: bool = False) -> list[ArchSupport]:
    """
    Try to install the extra architectures into the main registry.

    If `ignore_duplicates` is True, the function will not raise an error
    if the installation fails due to any of the architectures having already
    been installed (but they won't be replaced by ones from this package).
    """
    return MAIN_REGISTRY.add(*EXTRA_REGISTRY, ignore_duplicates=ignore_duplicates)


__all__ = [
    "EXTRA_REGISTRY",
    "install",
]
