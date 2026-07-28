"""Baseline transformer modules retained for controls and training infrastructure."""

# ============================================================
# Causal multi-head attention baseline (not Kimi Gated MLA)
# ============================================================

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer_modules.rope import RotaryEmbedding

# ============================================================
# CONFIG
# ============================================================

@dataclass
class CausalMHAConfig:
    d_model: int
    n_heads: int

    head_dim: Optional[int] = None

    attention_dropout: float = 0.0
    residual_dropout: float = 0.0

    use_bias: bool = False

    use_rope: bool = True
    rope_theta: float = 10000.0
    rotary_dim: Optional[int] = None

    max_seq_len: int = 1024
    init_std: float = 0.02

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {self.n_heads}")

        if self.head_dim is None:
            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    "If head_dim is None, d_model must be divisible by n_heads. "
                    f"Got d_model={self.d_model}, n_heads={self.n_heads}"
                )
            head_dim = self.d_model // self.n_heads
        else:
            head_dim = self.head_dim

        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")

        inner_dim = self.n_heads * head_dim

        # Baseline MHA: keep merge simple.
        if inner_dim != self.d_model:
            raise ValueError(
                "For baseline CausalMHA, n_heads * head_dim must equal d_model. "
                f"Got n_heads={self.n_heads}, head_dim={head_dim}, "
                f"inner_dim={inner_dim}, d_model={self.d_model}"
            )

        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError(
                "attention_dropout must satisfy 0 <= attention_dropout < 1, "
                f"got {self.attention_dropout}"
            )

        if not (0.0 <= self.residual_dropout < 1.0):
            raise ValueError(
                "residual_dropout must satisfy 0 <= residual_dropout < 1, "
                f"got {self.residual_dropout}"
            )

        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be > 0, got {self.rope_theta}")

        if self.rotary_dim is not None:
            if self.rotary_dim <= 0:
                raise ValueError(
                    f"rotary_dim must be > 0 when provided, got {self.rotary_dim}"
                )

            if self.rotary_dim > head_dim:
                raise ValueError(
                    f"rotary_dim must be <= head_dim. "
                    f"Got rotary_dim={self.rotary_dim}, head_dim={head_dim}"
                )

            if self.rotary_dim % 2 != 0:
                raise ValueError(
                    f"rotary_dim must be even, got {self.rotary_dim}"
                )


# ============================================================
# CAUSAL MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Baseline causal multi-head self-attention.

    Input:
        x: [B, T, d_model]

    Optional:
        attention_mask: [B, T]
            1 = valid token
            0 = padding / invalid key token

        position_ids:
            None, [T], or [B, T]

        start_pos:
            offset used by RoPE when position_ids is None

        need_weights:
            if True, returns attention weights.

    Output:
        out: [B, T, d_model]

        if need_weights=True:
            out, attn_weights
            attn_weights: [B, n_heads, T, T]
    """

    def __init__(self, config: CausalMHAConfig):
        super().__init__()

        config.validate()

        self.config = config

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else config.d_model // config.n_heads)

        self.inner_dim = self.n_heads * self.head_dim
        self.max_seq_len = config.max_seq_len

        self.use_rope = config.use_rope

        self.q_proj = nn.Linear(
            self.d_model,
            self.inner_dim,
            bias=config.use_bias,)

        self.k_proj = nn.Linear(
            self.d_model,
            self.inner_dim,
            bias=config.use_bias, )

        self.v_proj = nn.Linear(
            self.d_model,
            self.inner_dim,
            bias=config.use_bias,)

        self.out_proj = nn.Linear(
            self.inner_dim,
            self.d_model,
            bias=config.use_bias,)

        if self.use_rope:
            self.rope = RotaryEmbedding(
                dim=self.head_dim,
                rotary_dim=config.rotary_dim,
                base=config.rope_theta,)

        else:
            self.rope = None

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize projections with Normal(0, init_std).
        Biases, if present, are initialized to zero.
        """
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _shape_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert projected tensor from:

            [B, T, inner_dim]

        to:

            [B, T, n_heads, head_dim]
        """
        B, T, _ = x.shape

        return x.view(B, T, self.n_heads, self.head_dim)

    def _build_causal_mask(
        self,
        seq_len: int,
        device: torch.device) -> torch.Tensor:

        """
        Build causal future mask.

        Returns:
            causal_mask: [T, T]
            True means masked / forbidden.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,)

    def _validate_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        seq_len: int) -> torch.Tensor:
        """
        Validate and return attention_mask.

        Expected:
            attention_mask: [B, T]
            1 = valid
            0 = pad / invalid
        """
        if attention_mask.dim() != 2:
            raise ValueError(
                f"attention_mask must have shape [B, T], "
                f"got {tuple(attention_mask.shape)}")

        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attention_mask must have shape {(batch_size, seq_len)}, "
                f"got {tuple(attention_mask.shape)}")

        return attention_mask

    def _safe_masked_softmax(
      self,
      scores: torch.Tensor,
      allowed_mask: torch.Tensor,
      dim: int = -1) -> torch.Tensor:
      """
      Safe masked softmax.

      Args:
          scores:
              Attention scores [B, H, T, T].

          allowed_mask:
              Boolean mask broadcastable to scores.
              True  = allowed attention position.
              False = masked / forbidden position.

          dim:
              Softmax dimension.

      Returns:
          attn_weights:
              Same shape as scores.
              Rows with at least one valid key sum to 1.
              Rows with no valid keys are exactly zero.
      """

      if allowed_mask.dtype != torch.bool:
          allowed_mask = allowed_mask.bool()

      mask_value = torch.finfo(scores.dtype).min

      masked_scores = scores.masked_fill(~allowed_mask, mask_value)

      # Softmax in fp32 for numerical stability
      weights = F.softmax(masked_scores.float(), dim=dim).to(dtype=scores.dtype)

      # Remove any mass assigned to masked positions.
      weights = weights * allowed_mask.to(dtype=weights.dtype)

      # Renormalize only rows with at least one allowed key.
      denom = weights.sum(dim=dim, keepdim=True)

      weights = torch.where(
          denom > 0,
          weights / denom.clamp_min(torch.finfo(weights.dtype).tiny),
          torch.zeros_like(weights))

      return weights


    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        need_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:
                Hidden states [B, T, d_model].

            attention_mask:
                Optional mask [B, T].
                1 = valid token.
                0 = padding token.

            position_ids:
                Optional positions for RoPE.
                None, [T], or [B, T].

            start_pos:
                RoPE offset used when position_ids is None.

            need_weights:
                Whether to return attention weights.

        Returns:
            out:
                [B, T, d_model]

            optionally:
                attn_weights [B, n_heads, T, T]
        """

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------
        if x.dim() != 3:
            raise ValueError(
                f"MultiHeadSelfAttention expects x with shape [B, T, d_model], "
                f"got {tuple(x.shape)}")

        B, T, C = x.shape

        if C != self.d_model:
            raise ValueError(
                f"Expected x.shape[-1] == d_model={self.d_model}, got {C}")


        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}")


        if attention_mask is not None:
            attention_mask = self._validate_attention_mask(
                attention_mask=attention_mask,
                batch_size=B,
                seq_len=T,)


        # ----------------------------------------------------
        # QKV projections
        # ----------------------------------------------------
        q = self.q_proj(x)  # [B, T, inner_dim]
        k = self.k_proj(x)  # [B, T, inner_dim]
        v = self.v_proj(x)  # [B, T, inner_dim]

        q = self._shape_projection(q)  # [B, T, H, Dh]
        k = self._shape_projection(k)  # [B, T, H, Dh]
        v = self._shape_projection(v)  # [B, T, H, Dh]

        # ----------------------------------------------------
        # RoPE on q/k only
        # ----------------------------------------------------
        if self.rope is not None:
            q = self.rope(
                q,
                position_ids=position_ids,
                start_pos=start_pos,)

            k = self.rope(
                k,
                position_ids=position_ids,
                start_pos=start_pos,)

        # ----------------------------------------------------
        # Transpose for attention scores
        # ----------------------------------------------------
        q = q.transpose(1, 2)  # [B, H, T, Dh]
        k = k.transpose(1, 2)  # [B, H, T, Dh]
        v = v.transpose(1, 2)  # [B, H, T, Dh]

        # ----------------------------------------------------
        # Scaled dot-product attention scores
        # ----------------------------------------------------
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.head_dim)


        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------
        causal_mask = self._build_causal_mask(
            seq_len=T,
            device=x.device) # [T, T]

        allowed_mask = ~causal_mask[None, None, :, :]

        # --------------------------------------------------
        # Optional key padding mask
        # ----------------------------------------------------
        if attention_mask is not None:
            key_padding_mask = attention_mask[:, None, None, :].to(
                device=x.device,
                dtype=torch.bool) # [B, 1, 1, T]

            allowed_mask = allowed_mask & key_padding_mask

        # ----------------------------------------------------
        # Softmax attention weights
        # ----------------------------------------------------
        attn_weights = self._safe_masked_softmax(
            attn_scores,
            allowed_mask=allowed_mask,
            dim=-1,
        )

        attn_weights = self.attention_dropout(attn_weights)

        # ----------------------------------------------------
        # Weighted sum
        # ----------------------------------------------------
        context = torch.matmul(attn_weights, v) # [B, H, T, Dh]

        # ----------------------------------------------------
        # Merge heads
        # ----------------------------------------------------
        context = context.transpose(1, 2).contiguous()# [B, T, H, Dh]
        context = context.view(B, T, self.inner_dim) # [B, T, inner_dim]

        # ----------------------------------------------------
        # Output projection + residual dropout
        # ----------------------------------------------------
        out = self.out_proj(context)
        out = self.residual_dropout(out)

        if need_weights:
            return out, attn_weights

        return out

    def forward_decode(
        self,
        x_t: torch.Tensor,
        cache,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        if x_t.dim() != 3 or x_t.shape[1] != 1:
            raise ValueError(f"forward_decode expects x_t [B,1,D], got {tuple(x_t.shape)}")

        B, T, C = x_t.shape
        if C != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {C}")

        q = self._shape_projection(self.q_proj(x_t))
        k = self._shape_projection(self.k_proj(x_t))
        v = self._shape_projection(self.v_proj(x_t))

        if self.rope is not None:
            q = self.rope(q, position_ids=position_ids)
            k = self.rope(k, position_ids=position_ids)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cache.append(k, v, position_ids)
        k_all, v_all = cache.get_kv()

        attn_scores = torch.matmul(q, k_all.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.shape[0] != B:
                raise ValueError(
                    "decode attention_mask must have shape [B,T_cache], "
                    f"got {tuple(attention_mask.shape)}"
                )
            key_padding_mask = attention_mask[:, None, None, :].to(
                device=x_t.device,
                dtype=torch.bool,
            )
            attn_scores = attn_scores.masked_fill(
                ~key_padding_mask,
                torch.finfo(attn_scores.dtype).min,
            )

        attn_weights = F.softmax(attn_scores.float(), dim=-1).to(dtype=attn_scores.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        context = torch.matmul(attn_weights, v_all)
        context = context.transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        out = self.out_proj(context)
        out = self.residual_dropout(out)

        aux = {"attn_weights": attn_weights} if need_weights else {}
        return out, cache, aux


CausalMultiHeadAttention = MultiHeadSelfAttention
