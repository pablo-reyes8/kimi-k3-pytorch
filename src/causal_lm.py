"""Phase 0 causal Transformer baseline.

This is an integration control, not a paper-faithful Kimi-K3 backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .outputs import CausalLMOutput
from .transformer_modules import (
    BaselineTransformer,
    EmbeddingConfig,
    RMSNorm,
    TokenEmbedding,
    TransformerBlockConfig,
)


@dataclass
class BaselineCausalLMConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    mlp_hidden_dim: Optional[int] = None
    max_seq_len: int = 128
    pad_token_id: Optional[int] = None
    tie_embeddings: bool = True
    use_rope: bool = True
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    init_std: float = 0.02

    def validate(self) -> None:
        if self.vocab_size <= 0 or self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("vocab_size, d_model, and n_layers must be positive")
        if self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("n_heads must divide d_model")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if self.pad_token_id is not None and not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside the vocabulary")

    def embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            pad_token_id=self.pad_token_id,
            max_seq_len=self.max_seq_len,
            embedding_dropout=self.dropout,
            init_std=self.init_std,
            tie_word_embeddings=self.tie_embeddings,
        )

    def block_config(self) -> TransformerBlockConfig:
        return TransformerBlockConfig(
            d_model=self.d_model,
            rms_norm_eps=self.rms_norm_eps,
            n_heads=self.n_heads,
            attention_dropout=self.dropout,
            residual_dropout=self.dropout,
            use_rope=self.use_rope,
            rope_theta=self.rope_theta,
            max_seq_len=self.max_seq_len,
            mlp_hidden_dim=self.mlp_hidden_dim,
            mlp_dropout=self.dropout,
            init_std=self.init_std,
        )


class BaselineCausalLM(nn.Module):
    """MHA + RoPE + standard residuals + dense SwiGLU control model."""

    def __init__(self, config: BaselineCausalLMConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = TokenEmbedding(config.embedding_config())
        self.backbone = BaselineTransformer(config.block_config(), config.n_layers)
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=config.init_std)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward_embeddings(
        self,
        hidden_states: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"hidden_states must be [B,T,{self.config.d_model}], "
                f"got {tuple(hidden_states.shape)}"
            )
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
            raise ValueError("attention_mask must have shape [B,T]")

        hidden_states = self.backbone(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.lm_head(self.final_norm(hidden_states))

        loss = None
        if labels is not None:
            if labels.shape != (batch_size, seq_len):
                raise ValueError("labels must have shape [B,T]")
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=self.config.pad_token_id
                if self.config.pad_token_id is not None
                else -100,
            )
        return CausalLMOutput(logits=logits, loss=loss)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,T]")
        if attention_mask is None and self.config.pad_token_id is not None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        return self.forward_embeddings(
            self.embedding(input_ids),
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = (
            (p for p in self.parameters() if p.requires_grad)
            if trainable_only
            else self.parameters()
        )
        return sum(p.numel() for p in parameters)
