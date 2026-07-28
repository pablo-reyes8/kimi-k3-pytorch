"""Architecture-neutral stack used as the Phase 0 control model."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .baseline_block import BaselineTransformerBlock, TransformerBlockConfig


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerBlockConfig, n_layers: int):
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")
        self.layers = nn.ModuleList(
            BaselineTransformerBlock(config) for _ in range(n_layers)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return hidden_states
