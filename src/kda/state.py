"""Kimi Delta Attention operators, projections, states, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.kimi_primitives import ShortConvState


@dataclass
class KDAState:
    """Incremental KDA state containing recurrence and short-convolution buffers."""

    recurrent_state: torch.Tensor
    q_conv_state: ShortConvState
    k_conv_state: ShortConvState
    v_conv_state: ShortConvState
    sequence_offset: torch.Tensor

    def __post_init__(self) -> None:
        if self.recurrent_state.ndim != 4:
            raise ValueError("recurrent_state must have shape [B,H,K,V]")
        batch = self.recurrent_state.shape[0]
        if self.sequence_offset.shape != (batch,):
            raise ValueError(
                f"sequence_offset must have shape {(batch,)}, "
                f"got {tuple(self.sequence_offset.shape)}"
            )
        if self.sequence_offset.dtype != torch.long:
            raise TypeError("sequence_offset must use torch.long")
        conv_batches = (
            self.q_conv_state.buffer.shape[0],
            self.k_conv_state.buffer.shape[0],
            self.v_conv_state.buffer.shape[0],
        )
        if conv_batches != (batch, batch, batch):
            raise ValueError("all ShortConv states must match recurrent batch size")

    def clone(self) -> "KDAState":
        return KDAState(
            recurrent_state=self.recurrent_state.clone(),
            q_conv_state=self.q_conv_state.clone(),
            k_conv_state=self.k_conv_state.clone(),
            v_conv_state=self.v_conv_state.clone(),
            sequence_offset=self.sequence_offset.clone(),
        )

    def reorder(self, indices: torch.Tensor) -> "KDAState":
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a rank-1 torch.long tensor")
        if indices.device != self.recurrent_state.device:
            raise ValueError("indices and state must be on the same device")
        return KDAState(
            recurrent_state=self.recurrent_state.index_select(0, indices),
            q_conv_state=ShortConvState(
                self.q_conv_state.buffer.index_select(0, indices)
            ),
            k_conv_state=ShortConvState(
                self.k_conv_state.buffer.index_select(0, indices)
            ),
            v_conv_state=ShortConvState(
                self.v_conv_state.buffer.index_select(0, indices)
            ),
            sequence_offset=self.sequence_offset.index_select(0, indices),
        )
