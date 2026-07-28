"""Configuration and multimodal integration helpers used by the KimiK3 orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.hybrid_backbone import HybridBackboneCache
from src.mtp import MTPDiagnostics
from src.loss import KimiPretrainingLossOutput, MTPLossOutput, TokenCrossEntropyOutput
from src.vision import VisionEncoderOutput


@dataclass
class MultimodalMetadata:
    """Counts and token spans describing multimodal embedding replacement."""

    image_counts: torch.Tensor | None = None
    video_counts: torch.Tensor | None = None
    image_token_counts: torch.Tensor | None = None
    video_token_counts: torch.Tensor | None = None
    image_positions: torch.Tensor | None = None
    video_positions: torch.Tensor | None = None


@dataclass
class KimiK3VisionOutput:
    """Projected visual tokens and metadata produced before text composition."""

    images: VisionEncoderOutput | None = None
    videos: VisionEncoderOutput | None = None


@dataclass
class KimiK3Output:
    """Architecture output with an optional modular pretraining objective."""

    logits: torch.Tensor
    last_hidden_state: torch.Tensor
    cache: HybridBackboneCache | None = None
    mtp_logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    backbone_diagnostics: dict[str, object] | None = None
    attnres_diagnostics: object | None = None
    mtp_diagnostics: MTPDiagnostics | None = None
    vision_outputs: KimiK3VisionOutput | None = None
    multimodal_metadata: MultimodalMetadata | None = None
    loss: torch.Tensor | None = None
    ntp_loss: TokenCrossEntropyOutput | None = None
    mtp_loss: MTPLossOutput | None = None
    loss_output: KimiPretrainingLossOutput | None = None

    def to_tuple(self) -> tuple:
        return (
            self.logits,
            self.cache,
            self.hidden_states,
            self.mtp_logits,
        )


@dataclass(frozen=True)
class ParameterReport:
    """Unique parameter counts split by major KimiK3 subsystem."""

    total: int
    trainable: int
    embeddings: int
    vision: int
    backbone: int
    lm_head_unique: int
    mtp: int
