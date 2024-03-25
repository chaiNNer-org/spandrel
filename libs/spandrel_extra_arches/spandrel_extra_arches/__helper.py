from spandrel import ArchRegistry, ArchSupport

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
