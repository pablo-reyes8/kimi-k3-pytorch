from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import GatedMLAConfig


@dataclass
class LatentKVOutput:
    latent_kv: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


class LatentKVProjection(nn.Module):
    """Compress hidden states once and reconstruct both K and V from that latent."""

    def __init__(self, config: GatedMLAConfig):
        super().__init__()
        self.config = config
        self.compression = nn.Linear(
            config.d_model,
            config.kv_latent_dim,
            bias=config.projection_bias,
        )
        self.key_up = nn.Linear(
            config.kv_latent_dim,
            config.query_width,
            bias=config.projection_bias,
        )
        self.value_up = nn.Linear(
            config.kv_latent_dim,
            config.value_width,
            bias=config.projection_bias,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (self.compression, self.key_up, self.value_up):
            nn.init.normal_(
                projection.weight, mean=0.0, std=self.config.init_std
            )
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def compress(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compression(hidden_states)

    def reconstruct(
        self, latent_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            latent_kv.ndim != 3
            or latent_kv.shape[-1] != self.config.kv_latent_dim
        ):
            raise ValueError(
                "latent_kv must have shape "
                f"[B,T,{self.config.kv_latent_dim}]"
            )
        batch, tokens, _ = latent_kv.shape
        key = self.key_up(latent_kv).reshape(
            batch, tokens, self.config.num_heads, self.config.q_head_dim
        )
        value = self.value_up(latent_kv).reshape(
            batch, tokens, self.config.num_heads, self.config.v_head_dim
        )
        return key, value

    def forward(self, hidden_states: torch.Tensor) -> LatentKVOutput:
        latent = self.compress(hidden_states)
        key, value = self.reconstruct(latent)
        return LatentKVOutput(latent, key, value)
