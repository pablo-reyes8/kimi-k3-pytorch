"""Typed public output of autoregressive Kimi generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.hybrid_backbone import HybridBackboneCache


@dataclass
class GenerationOutput:
    prompt_ids: torch.Tensor
    sequences: torch.Tensor
    generated_ids: torch.Tensor
    text: str | list[str] | None
    completion_text: str | list[str] | None
    finish_reason: str
    cache: HybridBackboneCache | None
    cache_stats: dict[str, Any]
    scores: tuple[torch.Tensor, ...] | None
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    total_seconds: float
    tokens_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


__all__ = ["GenerationOutput"]
