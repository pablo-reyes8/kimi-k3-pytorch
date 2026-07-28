"""Reusable neural-network primitives shared by Kimi attention implementations."""

import torch
import torch.nn as nn


class HeadwiseRMSNorm(nn.Module):
    """RMS-normalize canonical ``[B,T,H,Dh]`` tensors over ``Dh`` only."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        per_head_affine: bool = True,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be > 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = float(eps)
        self.elementwise_affine = elementwise_affine
        self.per_head_affine = per_head_affine
        if elementwise_affine:
            shape = (num_heads, head_dim) if per_head_affine else (head_dim,)
            self.weight = nn.Parameter(
                torch.ones(shape, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"x must have shape [B,T,H,Dh], got {tuple(x.shape)}"
            )
        if x.shape[2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                f"expected head axes {(self.num_heads, self.head_dim)}, "
                f"got {tuple(x.shape[2:])}"
            )
        original_dtype = x.dtype
        accumulation = (
            x.float()
            if x.dtype in (torch.float16, torch.bfloat16)
            else x
        )
        normalized = accumulation * torch.rsqrt(
            accumulation.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = normalized.to(original_dtype)
        if self.weight is not None:
            weight = self.weight.to(dtype=original_dtype)
            normalized = normalized * (
                weight[None, None] if self.per_head_affine
                else weight[None, None, None]
            )
        return normalized
