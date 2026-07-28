"""Multi-token prediction components used as an optional KimiK3 output head."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.hybrid_backbone import HybridBackboneCache

from .alignment import MTPTrainingView


@dataclass
class MTPDiagnostics:
    """Shape and masking diagnostics collected by the MTP head."""

    valid_token_count: torch.Tensor
    token_accuracy: torch.Tensor
    mean_logit_entropy: torch.Tensor
    mean_hidden_norm: torch.Tensor
    block: dict[str, object] | None = None


@dataclass
class KimiMTPOutput:
    """Teacher-forced MTP logits, hidden states, masks, and diagnostics."""

    logits: torch.Tensor | None
    loss: torch.Tensor | None
    hidden_states: torch.Tensor | None
    training_view: MTPTrainingView
    diagnostics: MTPDiagnostics | None = None


@dataclass
class MTPDraftOutput:
    """Single-step speculative token proposal returned during inference."""

    logits: torch.Tensor
    cache: HybridBackboneCache | None
    hidden_states: torch.Tensor
