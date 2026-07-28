from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import GatedMLAConfig
from .latent_kv import LatentKVProjection


@dataclass
class MLAProjectionOutput:
    query: torch.Tensor
    latent_kv: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


class MLAProjections(nn.Module):
    def __init__(self, config: GatedMLAConfig):
        super().__init__()
        self.config = config
        self.query = nn.Linear(
            config.d_model,
            config.query_width,
            bias=config.projection_bias,
        )
        self.latent_kv = LatentKVProjection(config)
        nn.init.normal_(self.query.weight, mean=0.0, std=config.init_std)
        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)

    def project_queries(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        return self.query(hidden_states).reshape(
            batch, tokens, self.config.num_heads, self.config.q_head_dim
        )

    def compress_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.latent_kv.compress(hidden_states)

    def reconstruct_kv(
        self, latent_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.latent_kv.reconstruct(latent_kv)

    def forward(self, hidden_states: torch.Tensor) -> MLAProjectionOutput:
        query = self.project_queries(hidden_states)
        latent = self.compress_kv(hidden_states)
        key, value = self.reconstruct_kv(latent)
        return MLAProjectionOutput(query, latent, key, value)
