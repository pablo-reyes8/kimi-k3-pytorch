from .config import KimiOptimizerConfig
from .hybrid import (
    KimiHybridOptimizer,
    OptimizerStepReport,
    build_kimi_optimizer,
)
from .muon import KimiMuon
from .newton_schulz import match_update_rms, zeropower_via_newton_schulz
from .parameter_registry import (
    MatrixParameterSpec,
    ParameterAssignmentReport,
    build_parameter_registry,
)
from .per_head_muon import (
    HeadMatrixLayout,
    merge_head_matrix,
    per_head_orthogonalize,
    split_head_matrix,
)
from .qk_clip import QKClipController, QKClipReport

__all__ = [
    "HeadMatrixLayout",
    "KimiHybridOptimizer",
    "KimiMuon",
    "KimiOptimizerConfig",
    "MatrixParameterSpec",
    "OptimizerStepReport",
    "ParameterAssignmentReport",
    "QKClipController",
    "QKClipReport",
    "build_kimi_optimizer",
    "build_parameter_registry",
    "match_update_rms",
    "merge_head_matrix",
    "per_head_orthogonalize",
    "split_head_matrix",
    "zeropower_via_newton_schulz",
]
