"""Multi-token prediction components used as an optional KimiK3 output head."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.transformer_modules.rms_norm import RMSNorm


class KimiMTPFusion(nn.Module):
    """Normalize, concatenate and project h[t] with embedding(x[t+1])."""

    def __init__(self, d_model: int, eps: float = 1e-6, init_std: float = 0.02):
        super().__init__()
        self.d_model = d_model
        self.hidden_norm = RMSNorm(d_model, eps=eps)
        self.future_embedding_norm = RMSNorm(d_model, eps=eps)
        self.projection = nn.Linear(2 * d_model, d_model, bias=False)
        nn.init.normal_(self.projection.weight, mean=0.0, std=init_std)

    def forward(
        self,
        source_hidden: torch.Tensor,
        future_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        expected = source_hidden.shape
        if source_hidden.ndim != 3 or expected[-1] != self.d_model:
            raise ValueError(
                f"source_hidden must have shape [B,T,{self.d_model}]"
            )
        if future_embeddings.shape != expected:
            raise ValueError(
                "future_embeddings must have the same shape as source_hidden"
            )
        if source_hidden.device != future_embeddings.device:
            raise ValueError("fusion inputs must share device")
        if source_hidden.dtype != future_embeddings.dtype:
            raise TypeError("fusion inputs must share dtype")
        normalized = torch.cat(
            (
                self.hidden_norm(source_hidden),
                self.future_embedding_norm(future_embeddings),
            ),
            dim=-1,
        )
        return self.projection(normalized)
