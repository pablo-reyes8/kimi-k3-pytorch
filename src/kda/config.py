"""Kimi Delta Attention operators, projections, states, and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class KDAConfig:
    """Configure Kimi Delta Attention dimensions, kernels, and numerics."""

    d_model: int
    num_heads: int
    key_head_dim: int
    value_head_dim: int
    short_conv_kernel_size: int = 4
    decay_rank: int | None = None
    g_min: float = -5.0
    chunk_size: int = 64
    secondary_tile_size: int = 16
    eps: float = 1e-6
    projection_bias: bool = False
    short_conv_bias: bool = False
    beta_bias: bool = False
    output_gate_bias: bool = False
    output_bias: bool = False
    accumulate_state_in_fp32: bool = True
    decay_initializer: Literal["official_fla", "zeros"] = "official_fla"
    init_std: float = 0.02

    def __post_init__(self) -> None:
        dimensions = {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "key_head_dim": self.key_head_dim,
            "value_head_dim": self.value_head_dim,
            "short_conv_kernel_size": self.short_conv_kernel_size,
            "chunk_size": self.chunk_size,
            "secondary_tile_size": self.secondary_tile_size,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.d_model != self.num_heads * self.value_head_dim:
            raise ValueError(
                "d_model must equal num_heads * value_head_dim, "
                f"got {self.d_model} != {self.num_heads} * {self.value_head_dim}"
            )
        if self.decay_rank is not None and self.decay_rank <= 0:
            raise ValueError("decay_rank must be > 0 when provided")
        if self.g_min >= 0:
            raise ValueError(f"g_min must be negative, got {self.g_min}")
        if self.secondary_tile_size > self.chunk_size:
            raise ValueError("secondary_tile_size must be <= chunk_size")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if self.init_std <= 0:
            raise ValueError("init_std must be > 0")
        if self.decay_initializer not in ("official_fla", "zeros"):
            raise ValueError(
                "decay_initializer must be 'official_fla' or 'zeros'"
            )

    @property
    def resolved_decay_rank(self) -> int:
        return self.value_head_dim if self.decay_rank is None else self.decay_rank

    @property
    def key_width(self) -> int:
        return self.num_heads * self.key_head_dim

    @property
    def value_width(self) -> int:
        return self.num_heads * self.value_head_dim

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "KDAConfig":
        return cls(**values)
