from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MLACache:
    """Primary MLA state: compressed KV only, never reconstructed full K/V."""

    latent_kv: torch.Tensor
    attention_mask: torch.Tensor | None
    sequence_offset: torch.Tensor

    def __post_init__(self) -> None:
        if self.latent_kv.ndim != 3:
            raise ValueError("latent_kv must have shape [B,T,L]")
        batch, tokens, _ = self.latent_kv.shape
        if self.attention_mask is not None:
            if self.attention_mask.shape != (batch, tokens):
                raise ValueError(
                    f"cache attention_mask must have shape {(batch, tokens)}"
                )
            if self.attention_mask.dtype != torch.bool:
                raise TypeError("cache attention_mask must be boolean")
            if tokens > 1 and torch.any(
                (~self.attention_mask[:, :-1]) & self.attention_mask[:, 1:]
            ):
                raise ValueError("cache attention_mask must be right-padded")
        if self.sequence_offset.shape != (batch,):
            raise ValueError(
                f"sequence_offset must have shape {(batch,)}, "
                f"got {tuple(self.sequence_offset.shape)}"
            )
        if self.sequence_offset.dtype != torch.long:
            raise TypeError("sequence_offset must use torch.long")
        if self.sequence_offset.device != self.latent_kv.device:
            raise ValueError("sequence_offset and latent_kv must share device")
        lengths = (
            torch.full(
                (batch,), tokens, dtype=torch.long, device=self.latent_kv.device
            )
            if self.attention_mask is None
            else self.attention_mask.sum(dim=1).to(torch.long)
        )
        if not torch.equal(self.sequence_offset, lengths):
            raise ValueError(
                "sequence_offset must equal the number of valid cached tokens"
            )

    @classmethod
    def empty(
        cls,
        batch_size: int,
        latent_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "MLACache":
        if batch_size <= 0 or latent_dim <= 0:
            raise ValueError("batch_size and latent_dim must be > 0")
        return cls(
            latent_kv=torch.empty(
                batch_size, 0, latent_dim, device=device, dtype=dtype
            ),
            attention_mask=torch.empty(
                batch_size, 0, device=device, dtype=torch.bool
            ),
            sequence_offset=torch.zeros(
                batch_size, device=device, dtype=torch.long
            ),
        )

    @property
    def cache_length(self) -> int:
        return self.latent_kv.shape[1]

    @property
    def cache_elements(self) -> int:
        return self.latent_kv.numel()

    def clone(self) -> "MLACache":
        return MLACache(
            self.latent_kv.clone(),
            None if self.attention_mask is None else self.attention_mask.clone(),
            self.sequence_offset.clone(),
        )

    def reorder(self, indices: torch.Tensor) -> "MLACache":
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a rank-1 torch.long tensor")
        if indices.device != self.latent_kv.device:
            raise ValueError("indices and cache must share device")
        return MLACache(
            self.latent_kv.index_select(0, indices),
            None
            if self.attention_mask is None
            else self.attention_mask.index_select(0, indices),
            self.sequence_offset.index_select(0, indices),
        )

    def append(
        self,
        latent_kv: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> "MLACache":
        if latent_kv.ndim != 3:
            raise ValueError("new latent_kv must have shape [B,T,L]")
        batch, tokens, latent_dim = latent_kv.shape
        if (
            batch != self.latent_kv.shape[0]
            or latent_dim != self.latent_kv.shape[2]
        ):
            raise ValueError("new latent_kv batch and latent dim must match cache")
        if latent_kv.device != self.latent_kv.device:
            raise ValueError("new latent_kv and cache must share device")
        if latent_kv.dtype != self.latent_kv.dtype:
            raise TypeError("new latent_kv and cache must share dtype")
        new_mask = (
            torch.ones(batch, tokens, dtype=torch.bool, device=latent_kv.device)
            if attention_mask is None
            else attention_mask
        )
        if new_mask.shape != (batch, tokens) or new_mask.dtype != torch.bool:
            raise ValueError("new attention_mask must be boolean with shape [B,T]")
        old_mask = (
            torch.ones(
                batch,
                self.cache_length,
                dtype=torch.bool,
                device=latent_kv.device,
            )
            if self.attention_mask is None
            else self.attention_mask
        )
        sequences = [
            torch.cat(
                (self.latent_kv[index][old_mask[index]], latent_kv[index][new_mask[index]]),
                dim=0,
            )
            for index in range(batch)
        ]
        lengths = torch.tensor(
            [sequence.shape[0] for sequence in sequences],
            dtype=torch.long,
            device=latent_kv.device,
        )
        max_length = int(lengths.max().item()) if batch else 0
        packed = torch.stack(
            [
                F.pad(sequence, (0, 0, 0, max_length - sequence.shape[0]))
                for sequence in sequences
            ]
        )
        packed_mask = (
            torch.arange(max_length, device=latent_kv.device)[None, :]
            < lengths[:, None]
        )
        return MLACache(packed, packed_mask, lengths)
