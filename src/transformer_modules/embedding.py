"""Baseline transformer modules retained for controls and training infrastructure."""

# ============================================================
# Generic token embedding module adapted for Kimi-K3 Mini Phase 0
# Token identity only — no positional embeddings
# ============================================================

import math
from dataclasses import dataclass
from typing import Optional

import torch
from typing import Dict, Union
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

@dataclass
class EmbeddingConfig:
    vocab_size: int
    d_model: int

    pad_token_id: Optional[int] = None
    max_seq_len: int = 1024

    embedding_dropout: float = 0.0
    scale_embeddings: bool = False

    init_std: float = 0.02
    tie_word_embeddings: bool = True

    def validate(self) -> None:
        """
        Validate embedding configuration early.

        This prevents silent configuration bugs before constructing
        nn.Embedding or starting training.
        """
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")

        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")

        if not (0.0 <= self.embedding_dropout < 1.0):
            raise ValueError(
                "embedding_dropout must satisfy 0 <= embedding_dropout < 1, "
                f"got {self.embedding_dropout}")

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

        if self.pad_token_id is not None:
            if not (0 <= self.pad_token_id < self.vocab_size):
                raise ValueError(
                    "pad_token_id must satisfy 0 <= pad_token_id < vocab_size, "
                    f"got pad_token_id={self.pad_token_id}, "
                    f"vocab_size={self.vocab_size}")


# ============================================================
# TOKEN EMBEDDING
# ============================================================

class TokenEmbedding(nn.Module):
    """
    Token embedding module for causal language modeling.

    Responsibility:
        input_ids: [B, T] -> hidden_states: [B, T, d_model]

    This module intentionally does NOT include:
        - positional embeddings
        - RoPE
        - attention masks
        - RMSNorm
        - MoE
        - mHC
        - loss logic

    Positional information should be handled later inside attention,
    e.g. via RoPE or partial RoPE.
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__()

        config.validate()

        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.pad_token_id = config.pad_token_id
        self.max_seq_len = config.max_seq_len
        self.scale_embeddings = config.scale_embeddings

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,)

        self.dropout = nn.Dropout(config.embedding_dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize token embedding weights.

        Uses GPT-style normal initialization:
            weight ~ N(0, init_std)

        If pad_token_id is provided, its row is forced to zero.
        nn.Embedding(..., padding_idx=...) also prevents that row
        from receiving gradient updates.
        """
        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=self.config.init_std,)

        if self.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.pad_token_id].zero_()

    def forward(
          self,
          input_ids: Union[torch.Tensor, Dict[str, torch.Tensor]],) -> torch.Tensor:
          """
          Args:
              input_ids:
                  Either:
                    - Tensor [B, T]
                    - Dict with key "input_ids"

          Returns:
              hidden_states:
                  Float tensor [B, T, d_model]
          """

          # ----------------------------------------------------
          # Accept dict batches defensively
          # ----------------------------------------------------
          if isinstance(input_ids, dict):
              if "input_ids" not in input_ids:
                  raise KeyError(
                      "TokenEmbedding received a dict batch but it does not contain "
                      f"'input_ids'. Available keys: {list(input_ids.keys())}"
                  )

              input_ids = input_ids["input_ids"]

          # ----------------------------------------------------
          # Shape validation
          # ----------------------------------------------------
          if not torch.is_tensor(input_ids):
              raise TypeError(
                  "TokenEmbedding expects either a tensor [B, T] or a dict containing "
                  f"'input_ids'. Got type: {type(input_ids)}"
              )

          if input_ids.dim() != 2:
              raise ValueError(
                  f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}"
              )

          _, seq_len = input_ids.shape

          if seq_len > self.max_seq_len:
              raise ValueError(
                  f"Sequence length T={seq_len} exceeds max_seq_len={self.max_seq_len}"
              )

          # ----------------------------------------------------
          # Dtype validation
          # ----------------------------------------------------
          if input_ids.dtype not in (torch.long, torch.int64, torch.int32):
              raise TypeError(
                  "input_ids must contain integer token indices; "
                  f"got dtype={input_ids.dtype}"
              )

          if input_ids.dtype != torch.long:
              input_ids = input_ids.long()

          # ----------------------------------------------------
          # Range validation
          # ----------------------------------------------------
          if torch.any(input_ids < 0):
              min_id = int(input_ids.min().item())
              raise ValueError(
                  f"input_ids contain negative token ids. Minimum id found: {min_id}"
              )

          if torch.any(input_ids >= self.vocab_size):
              max_id = int(input_ids.max().item())
              raise ValueError(
                  "input_ids contain token ids >= vocab_size. "
                  f"Maximum id found: {max_id}, vocab_size={self.vocab_size}"
              )

          # ----------------------------------------------------
          # Token embedding lookup
          # ----------------------------------------------------
          hidden_states = self.token_embedding(input_ids)

          # ----------------------------------------------------
          # Optional scaling
          # ----------------------------------------------------
          if self.scale_embeddings:
              hidden_states = hidden_states * math.sqrt(self.d_model)

          # ----------------------------------------------------
          # Dropout
          # ----------------------------------------------------
          hidden_states = self.dropout(hidden_states)

          return hidden_states

    @property
    def weight(self) -> nn.Parameter:
        """
        Expose embedding weight for optional LM-head weight tying.

        Example in full model:
            lm_head.weight = embedding.weight
        """
        return self.token_embedding.weight
