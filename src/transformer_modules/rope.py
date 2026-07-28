import torch
import torch.nn as nn
from typing import Optional

from src.transformer_modules.rope_utils import rotate_half
# ============================================================
# RotaryEmbedding
# ============================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding utility.

    Expected input:
        x: [B, T, H, D]

    where:
        B = batch size
        T = sequence length
        H = number of heads
        D = head_dim

    Supports:
        - full RoPE: rotary_dim == dim
        - partial RoPE: rotary_dim < dim
        - automatic positions with start_pos
        - explicit position_ids with shape [T]
        - explicit position_ids with shape [B, T]
        - negative positions
        - positions larger than max_seq_len

    This module does NOT implement:
        - attention
        - q/k/v projections
        - KV cache
        - RoPE scaling
        - learned positional embeddings
        - cos/sin caching
    """

    def __init__(
        self,
        dim: int,
        rotary_dim: Optional[int] = None,
        base: float = 10000.0):

        super().__init__()

        if rotary_dim is None:
            rotary_dim = dim

        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")

        if rotary_dim <= 0:
            raise ValueError(f"rotary_dim must be > 0, got {rotary_dim}")

        if rotary_dim > dim:
            raise ValueError(
                f"rotary_dim must be <= dim, got rotary_dim={rotary_dim}, dim={dim}"
            )

        if rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary_dim must be even, got rotary_dim={rotary_dim}"
            )

        if base <= 0:
            raise ValueError(f"base must be > 0, got {base}")

        self.dim = dim
        self.rotary_dim = rotary_dim
        self.base = base

        # Conceptually:
        # inv_freq[i] = 1 / base^(i / rotary_dim), for even rotary indices.
        #
        # torch.arange(0, rotary_dim, 2) gives:
        # 0, 2, 4, ...
        # so exponent becomes:
        # 0/rotary_dim, 2/rotary_dim, 4/rotary_dim, ...
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                / rotary_dim
            ))

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False)


    def _build_position_ids(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        position_ids: Optional[torch.Tensor],
        start_pos: int,) -> torch.Tensor:
        """
        Build or validate position_ids.

        Supported:
            None      -> [T]
            [T]       -> [T]
            [B, T]    -> [B, T]

        Negative positions are allowed.
        """

        if position_ids is None:
            return torch.arange(
                start_pos,
                start_pos + seq_len,
                device=device,
                dtype=torch.float32)

        if position_ids.device != device:
            position_ids = position_ids.to(device)

        if position_ids.dim() == 1:
            if position_ids.shape[0] != seq_len:
                raise ValueError(
                    f"position_ids with shape [T] must have length T={seq_len}, "
                    f"got {position_ids.shape[0]}")

            return position_ids.float()

        if position_ids.dim() == 2:
            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    "position_ids with shape [B, T] must match input batch/length. "
                    f"Expected {(batch_size, seq_len)}, got {tuple(position_ids.shape)}")

            return position_ids.float()

        raise ValueError(
            "position_ids must be None, shape [T], or shape [B, T], "
            f"got shape {tuple(position_ids.shape)}")

    def _build_cos_sin(
        self,
        position_ids: torch.Tensor,
        target_dtype: torch.dtype,
        device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build cos/sin tensors dynamically.

        If position_ids:
            [T]    -> cos/sin: [1, T, 1, rotary_dim]
            [B,T]  -> cos/sin: [B, T, 1, rotary_dim]
        """

        position_ids = position_ids.to(device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)

        # freqs:
        # [T, rotary_dim // 2] or [B, T, rotary_dim // 2]
        freqs = position_ids[..., None] * inv_freq

        # Expand from half dimension to full rotary_dim.
        # Shape:
        # [T, rotary_dim] or [B, T, rotary_dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = torch.cos(emb)
        sin = torch.sin(emb)

        if position_ids.dim() == 1:
            # [T, R] -> [1, T, 1, R]
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            # [B, T, R] -> [B, T, 1, R]
            cos = cos[:, :, None, :]
            sin = sin[:, :, None, :]

        cos = cos.to(dtype=target_dtype)
        sin = sin.to(dtype=target_dtype)

        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x:
                Attention tensor with shape [B, T, H, D].

            position_ids:
                Optional positions:
                    None
                    [T]
                    [B, T]

            start_pos:
                Offset used only when position_ids is None.

        Returns:
            y:
                Tensor with same shape, dtype, and device as x.
        """

        if x.dim() != 4:
            raise ValueError(
                f"RotaryEmbedding expects x with shape [B, T, H, D], "
                f"got {tuple(x.shape)}")

        batch_size, seq_len, _, head_dim = x.shape

        if head_dim != self.dim:
            raise ValueError(
                f"Expected x.shape[-1] == dim={self.dim}, got {head_dim}")

        original_dtype = x.dtype
        device = x.device

        position_ids = self._build_position_ids(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            position_ids=position_ids,
            start_pos=start_pos,)

        cos, sin = self._build_cos_sin(
            position_ids=position_ids,
            target_dtype=original_dtype,
            device=device)

        pass_dim = self.dim - self.rotary_dim

        if pass_dim > 0:
            x_pass = x[..., :pass_dim]
            x_rot = x[..., pass_dim:]
        else:
            x_pass = None
            x_rot = x

        x_rotated = (x_rot * cos) + (rotate_half(x_rot) * sin)

        if x_pass is not None:
            y = torch.cat((x_pass, x_rotated), dim=-1)
        else:
            y = x_rotated

        return y
