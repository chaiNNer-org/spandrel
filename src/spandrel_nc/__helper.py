from spandrel import ArchRegistry, ArchSupport

from .architectures import (
    MAT,
    CodeFormer,
    DDColor,
    SRFormer,
)

NC_REGISTRY = ArchRegistry()

NC_REGISTRY.add(
    ArchSupport.from_architecture(SRFormer.SRFormerArch()),
    ArchSupport.from_architecture(CodeFormer.CodeFormerArch()),
    ArchSupport.from_architecture(MAT.MATArch()),
    ArchSupport.from_architecture(DDColor.DDColorArch()),
)
