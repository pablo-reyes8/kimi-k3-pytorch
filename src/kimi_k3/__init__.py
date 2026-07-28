"""Configuration and multimodal integration helpers used by the KimiK3 orchestrator."""

from .config import (
    KimiK3Config,
    VisionConfig,
    VisionProjectorConfig,
    kimi_k3_canonical_config,
    kimi_k3_cpu_tiny_config,
)
from .multimodal_composer import VisualPlaceholderComposer
from .outputs import (
    KimiK3Output,
    KimiK3VisionOutput,
    MultimodalMetadata,
    ParameterReport,
)

__all__ = [
    "KimiK3Config",
    "KimiK3Output",
    "KimiK3VisionOutput",
    "MultimodalMetadata",
    "ParameterReport",
    "VisionConfig",
    "VisionProjectorConfig",
    "VisualPlaceholderComposer",
    "kimi_k3_canonical_config",
    "kimi_k3_cpu_tiny_config",
]
