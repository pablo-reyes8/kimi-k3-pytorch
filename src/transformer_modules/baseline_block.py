# ============================================================
# Standard pre-norm Transformer control block
# Pre-Norm Dense Transformer Block
# ============================================================

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn


from src.transformer_modules.rms_norm import RMSNorm
from src.transformer_modules.mha import CausalMHAConfig, MultiHeadSelfAttention
from src.transformer_modules.swiglu import SwiGLUFeedForward, SwiGLUMLPConfig
# ============================================================
# CONFIG
# ============================================================

@dataclass
class TransformerBlockConfig:
    d_model: int
    rms_norm_eps: float = 1e-6

    # Attention config
    n_heads: int = 4
    head_dim: Optional[int] = None
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    use_attention_bias: bool = False
    use_rope: bool = True
    rope_theta: float = 10000.0
    rotary_dim: Optional[int] = None
    max_seq_len: int = 1024

    # MLP config
    mlp_hidden_dim: Optional[int] = None
    mlp_expansion_factor: float = 4.0
    mlp_multiple_of: int = 1
    mlp_dropout: float = 0.0
    use_mlp_bias: bool = False

    # Initialization
    init_std: float = 0.02

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        if self.rms_norm_eps <= 0:
            raise ValueError(
                f"rms_norm_eps must be > 0, got {self.rms_norm_eps}"
            )

        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")

        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")

        # Validate attention by constructing its config and calling validate.
        attention_config = self.to_attention_config()
        attention_config.validate()

        # Validate MLP by constructing its config and calling validate.
        mlp_config = self.to_mlp_config()
        mlp_config.validate()

        if attention_config.d_model != self.d_model:
            raise ValueError(
                "attention_config.d_model must match block d_model. "
                f"Got {attention_config.d_model} vs {self.d_model}"
            )

        if mlp_config.d_model != self.d_model:
            raise ValueError(
                "mlp_config.d_model must match block d_model. "
                f"Got {mlp_config.d_model} vs {self.d_model}"
            )

    def to_attention_config(self) -> "CausalMHAConfig":
        return CausalMHAConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            attention_dropout=self.attention_dropout,
            residual_dropout=self.residual_dropout,
            use_bias=self.use_attention_bias,
            use_rope=self.use_rope,
            rope_theta=self.rope_theta,
            rotary_dim=self.rotary_dim,
            max_seq_len=self.max_seq_len,
            init_std=self.init_std)

    def to_mlp_config(self) -> "SwiGLUMLPConfig":
        return SwiGLUMLPConfig(
            d_model=self.d_model,
            hidden_dim=self.mlp_hidden_dim,
            expansion_factor=self.mlp_expansion_factor,
            multiple_of=self.mlp_multiple_of,
            dropout=self.mlp_dropout,
            use_bias=self.use_mlp_bias,
            init_std=self.init_std,)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class BaselineTransformerBlock(nn.Module):
    """
    Dense pre-norm causal Transformer block.

    Input:
        x: [B, T, d_model]

    Forward:
        x = x + attention(norm1(x))
        x = x + mlp(norm2(x))

    Output:
        x: [B, T, d_model]

    If need_weights=True:
        returns:
            x, {"attn_weights": attn_weights}

    This module intentionally does NOT include:
        - MoE
        - mHC
        - HCA
        - CSA
        - KV cache
        - gradient checkpointing
        - attention sink
        - query/key RMSNorm
    """

    def __init__(self, config: TransformerBlockConfig):
        super().__init__()

        config.validate()

        self.config = config
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len

        self.norm1 = RMSNorm(
            dim=config.d_model,
            eps=config.rms_norm_eps)

        self.attention = MultiHeadSelfAttention(
            config.to_attention_config())

        self.norm2 = RMSNorm(
            dim=config.d_model,
            eps=config.rms_norm_eps,)

        self.mlp = SwiGLUFeedForward(
            config.to_mlp_config())

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        need_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            x:
                Hidden states [B, T, d_model].

            attention_mask:
                Optional attention mask [B, T].
                1 = valid token.
                0 = padding token.

            position_ids:
                Optional RoPE position ids.
                None, [T], or [B, T].

            start_pos:
                RoPE offset when position_ids is None.

            need_weights:
                Whether to return attention weights in aux dict.

        Returns:
            x:
                Updated hidden states [B, T, d_model].

            optionally:
                x, {"attn_weights": attn_weights}
        """

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------
        if x.dim() != 3:
            raise ValueError(
                f"BaselineTransformerBlock expects x with shape [B, T, d_model], "
                f"got {tuple(x.shape)}"
            )

        B, T, C = x.shape

        if C != self.d_model:
            raise ValueError(
                f"Expected x.shape[-1] == d_model={self.d_model}, got {C}"
            )

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        # ----------------------------------------------------
        # Attention sublayer: pre-norm + residual
        # ----------------------------------------------------
        residual = x

        x_norm = self.norm1(x)

        attn_result = self.attention(
            x_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            start_pos=start_pos,
            need_weights=need_weights)

        if need_weights:
            attn_out, attn_weights = attn_result
        else:
            attn_out = attn_result
            attn_weights = None

        x = residual + attn_out

        # ----------------------------------------------------
        # MLP sublayer: pre-norm + residual
        # ----------------------------------------------------
        residual = x

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)

        x = residual + mlp_out

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------
        if need_weights:
            aux = {
                "attn_weights": attn_weights}
            return x, aux

        return x


TransformerBlock = BaselineTransformerBlock
