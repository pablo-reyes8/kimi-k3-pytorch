"""Attention-residual components for mixing hidden-state streams across model depth."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .utils import stack_sources


@dataclass
class FullAttentionResidualState:
    """Mutable hidden-state history used by full attention residuals."""

    sources: list[torch.Tensor]
    num_sublayer_outputs: int = 0

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("Full AttnRes state requires the embedding source")
        if len(self.sources) != 1 + self.num_sublayer_outputs:
            raise ValueError("Full AttnRes source count invariant violated")
        stack_sources(self.sources)

    @property
    def embedding(self) -> torch.Tensor:
        return self.sources[0]

    @property
    def source_elements(self) -> int:
        return sum(source.numel() for source in self.sources)

    def available_sources(self) -> torch.Tensor:
        return stack_sources(self.sources)

    def append_output(self, output: torch.Tensor) -> None:
        stack_sources([self.embedding, output])
        self.sources.append(output)
        self.num_sublayer_outputs += 1

    def with_appended(self, output: torch.Tensor) -> "FullAttentionResidualState":
        new_state = self.clone()
        new_state.append_output(output)
        return new_state

    def clone(self, detach: bool = False) -> "FullAttentionResidualState":
        clone = [
            source.detach().clone() if detach else source.clone()
            for source in self.sources
        ]
        return FullAttentionResidualState(clone, self.num_sublayer_outputs)
