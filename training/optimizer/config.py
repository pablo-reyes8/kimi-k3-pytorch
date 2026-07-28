"""Configuration for Kimi-style hybrid optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class KimiOptimizerConfig:
    kind: Literal[
        "adamw", "muon_adamw", "per_head_muon_adamw"
    ] = "per_head_muon_adamw"
    adamw_lr: float = 3e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    muon_lr: float | None = None
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_ns_eps: float = 1e-7
    muon_update_rms_scaling: bool = True
    weight_decay: float = 0.1
    muon_weight_decay: float | None = None
    per_head_qkv: bool = True
    qk_clip_enabled: bool = True
    qk_clip_threshold: float = 100.0
    qk_clip_eps: float = 1e-6
    qk_clip_every_steps: int = 1
    qk_clip_kda_experimental: bool = False
    fail_on_unclassified_matrix: bool = True

    def __post_init__(self) -> None:
        if self.kind not in {
            "adamw",
            "muon_adamw",
            "per_head_muon_adamw",
        }:
            raise ValueError("unknown optimizer kind")
        if self.adamw_lr <= 0 or self.adamw_eps <= 0:
            raise ValueError("AdamW learning rate and epsilon must be positive")
        if (
            len(self.adamw_betas) != 2
            or not all(0 <= value < 1 for value in self.adamw_betas)
        ):
            raise ValueError("adamw_betas must contain two values in [0, 1)")
        if self.muon_lr is not None and self.muon_lr <= 0:
            raise ValueError("muon_lr must be None or positive")
        if not 0 <= self.muon_momentum < 1:
            raise ValueError("muon_momentum must be in [0, 1)")
        if self.muon_ns_steps <= 0 or self.muon_ns_eps <= 0:
            raise ValueError("Newton-Schulz steps and epsilon must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.muon_weight_decay is not None and self.muon_weight_decay < 0:
            raise ValueError("muon_weight_decay must be None or non-negative")
        if self.qk_clip_threshold <= 0 or self.qk_clip_eps <= 0:
            raise ValueError("QK-Clip threshold and epsilon must be positive")
        if self.qk_clip_every_steps <= 0:
            raise ValueError("qk_clip_every_steps must be positive")
        if self.kind == "per_head_muon_adamw" and not self.per_head_qkv:
            raise ValueError("canonical Per-Head Muon requires per_head_qkv=True")

    @property
    def resolved_muon_lr(self) -> float:
        return self.adamw_lr if self.muon_lr is None else self.muon_lr

    @property
    def resolved_muon_weight_decay(self) -> float:
        return (
            self.weight_decay
            if self.muon_weight_decay is None
            else self.muon_weight_decay
        )

    def to_dict(self) -> dict:
        return asdict(self)
