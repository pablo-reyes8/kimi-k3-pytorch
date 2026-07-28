import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionSelfAttention(nn.Module):
    """Global multi-head self-attention with True=valid padding masks."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ):
        super().__init__()
        if dim <= 0 or num_heads <= 0:
            raise ValueError("dim and num_heads must be > 0")
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        for name, value in (
            ("attention_dropout", attention_dropout),
            ("projection_dropout", projection_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dim, dim, bias=proj_bias)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"x must have shape [B, N, {self.dim}], got {tuple(x.shape)}"
            )
        batch, count, _ = x.shape
        if padding_mask is not None:
            if padding_mask.shape != (batch, count):
                raise ValueError(
                    f"padding_mask must have shape {(batch, count)}, "
                    f"got {tuple(padding_mask.shape)}"
                )
            if padding_mask.dtype != torch.bool:
                raise TypeError("padding_mask must be boolean (True means valid)")

        qkv = self.qkv(x).reshape(
            batch, count, 3, self.num_heads, self.head_dim
        )
        q, k, value = qkv.permute(2, 0, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if padding_mask is not None:
            valid_keys = padding_mask[:, None, None, :]
            scores = scores.masked_fill(~valid_keys, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        if padding_mask is not None:
            probabilities = probabilities * valid_keys.to(probabilities.dtype)
            probabilities = probabilities / probabilities.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(probabilities.dtype).tiny)
        dropped = self.attention_dropout(probabilities)
        output = torch.matmul(dropped, value)
        output = output.transpose(1, 2).reshape(batch, count, self.dim)
        output = self.projection_dropout(self.projection(output))
        return output, probabilities if output_attentions else None

