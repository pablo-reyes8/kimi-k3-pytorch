"""Reusable neural-network primitives shared by Kimi attention implementations."""

import torch


def split_heads(hidden_states: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Convert ``[B,T,D]`` to ``[B,T,H,Dh]`` without permuting channels."""
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must have shape [B,T,D], got {tuple(hidden_states.shape)}"
        )
    if num_heads <= 0:
        raise ValueError(f"num_heads must be > 0, got {num_heads}")
    batch, tokens, width = hidden_states.shape
    if width % num_heads:
        raise ValueError(
            f"model width D={width} must be divisible by num_heads={num_heads}"
        )
    return hidden_states.reshape(batch, tokens, num_heads, width // num_heads)


def combine_heads(head_states: torch.Tensor) -> torch.Tensor:
    """Convert canonical ``[B,T,H,Dh]`` tensors to ``[B,T,D]``."""
    if head_states.ndim != 4:
        raise ValueError(
            f"head_states must have shape [B,T,H,Dh], got {tuple(head_states.shape)}"
        )
    batch, tokens, heads, head_dim = head_states.shape
    if heads <= 0 or head_dim <= 0:
        raise ValueError("head and head_dim axes must be non-empty")
    return head_states.reshape(batch, tokens, heads * head_dim)

