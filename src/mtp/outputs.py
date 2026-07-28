from __future__ import annotations

from dataclasses import dataclass

import torch

from src.hybrid_backbone import HybridBackboneCache

from .alignment import MTPTrainingView


@dataclass
class MTPDiagnostics:
    valid_token_count: torch.Tensor
    token_accuracy: torch.Tensor
    mean_logit_entropy: torch.Tensor
    mean_hidden_norm: torch.Tensor
    block: dict[str, object] | None = None


@dataclass
class KimiMTPOutput:
    logits: torch.Tensor | None
    loss: torch.Tensor | None
    hidden_states: torch.Tensor | None
    training_view: MTPTrainingView
    diagnostics: MTPDiagnostics | None = None


@dataclass
class MTPDraftOutput:
    logits: torch.Tensor
    cache: HybridBackboneCache | None
    hidden_states: torch.Tensor
