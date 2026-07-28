"""Reusable neural-network primitives shared by Kimi attention implementations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KDAProjectionOutput:
    """Typed KDA projections; leading dimensions are ``[B,T,...]``."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    beta: torch.Tensor
    alpha: torch.Tensor

    def __post_init__(self) -> None:
        values = (self.q, self.k, self.v, self.beta, self.alpha)
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("all KDA projections must be torch.Tensor instances")
        if self.q.shape != self.k.shape or self.q.shape != self.v.shape:
            raise ValueError("q, k, and v must have identical shapes")
        if self.q.ndim < 3:
            raise ValueError("q, k, and v must have at least [B,T,D] dimensions")
        leading = self.q.shape[:2]
        if self.beta.shape[:2] != leading or self.alpha.shape[:2] != leading:
            raise ValueError("beta and alpha must share q/k/v batch and token axes")


@dataclass
class AttentionModuleOutput:
    """Stable output contract for future attention implementations."""

    hidden_states: torch.Tensor
    state: object | None = None
    diagnostics: dict[str, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if self.hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [B,T,D], "
                f"got {tuple(self.hidden_states.shape)}"
            )
        if self.diagnostics is not None:
            if not isinstance(self.diagnostics, dict):
                raise TypeError("diagnostics must be a dict or None")
            if not all(
                isinstance(name, str) and isinstance(value, torch.Tensor)
                for name, value in self.diagnostics.items()
            ):
                raise TypeError("diagnostics must map strings to tensors")

