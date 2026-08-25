"""LayerMix: multi-layer probing for hallucination detection."""

from layermix.geometry import (
    cohens_d,
    mean_shift_direction,
    project_out,
    decompose_signal,
)
from layermix.probes import (
    full_probe,
    mean_shift_probe,
    shrinkage_lda_probe,
)
from layermix.layermix import LayerMix

__all__ = [
    "cohens_d",
    "mean_shift_direction",
    "project_out",
    "decompose_signal",
    "full_probe",
    "mean_shift_probe",
    "shrinkage_lda_probe",
    "LayerMix",
]
