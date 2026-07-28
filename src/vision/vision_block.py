"""MoonViT, hierarchical, and Swin vision encoder components."""

import torch
import torch.nn as nn

from .drop_path import DropPath
from .utils import build_vision_norm
from .vision_attention import VisionSelfAttention
from .vision_mlp import VisionMLP


class VisionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        mlp_bias: bool = False,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")
        self.norm1 = build_vision_norm(norm_type, dim, norm_eps)
        self.attention = VisionSelfAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = build_vision_norm(norm_type, dim, norm_eps)
        self.mlp = VisionMLP(
            dim, int(dim * mlp_ratio), dropout=dropout, bias=mlp_bias
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attention(
            self.norm1(x),
            padding_mask,
            output_attentions=output_attentions,
        )
        x = x + self.drop_path1(attended)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, weights

