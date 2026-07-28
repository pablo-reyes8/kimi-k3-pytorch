from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class GatedMLAConfig:
    """Configuration for Kimi K3's global, NoPE Gated MLA operator."""

    d_model: int
    num_heads: int
    q_head_dim: int
    v_head_dim: int
    kv_latent_dim: int
    attention_dropout: float = 0.0
    output_dropout: float = 0.0
    projection_bias: bool = False
    output_gate_bias: bool = False
    output_bias: bool = False
    use_nope: bool = True
    keep_attention_output_fp32: bool = True
    use_sdpa: bool = True
    attention_backend: Literal["auto", "manual", "sdpa"] = "auto"
    init_std: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "num_heads",
            "q_head_dim",
            "v_head_dim",
            "kv_latent_dim",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.d_model != self.num_heads * self.v_head_dim:
            raise ValueError(
                "d_model must equal num_heads * v_head_dim, "
                f"got {self.d_model} != {self.num_heads} * {self.v_head_dim}"
            )
        full_kv_width = self.num_heads * (
            self.q_head_dim + self.v_head_dim
        )
        if self.kv_latent_dim > full_kv_width:
            raise ValueError(
                "kv_latent_dim must not exceed the uncompressed per-token KV "
                f"width ({full_kv_width}), got {self.kv_latent_dim}"
            )
        for name in ("attention_dropout", "output_dropout"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must satisfy 0 <= p < 1, got {value}")
        if not self.use_nope:
            raise ValueError("Kimi K3 Gated MLA requires use_nope=True")
        if self.attention_backend not in ("auto", "manual", "sdpa"):
            raise ValueError(
                "attention_backend must be 'auto', 'manual', or 'sdpa'"
            )
        if self.attention_backend == "sdpa" and not self.use_sdpa:
            raise ValueError(
                "attention_backend='sdpa' is incompatible with use_sdpa=False"
            )
        if self.init_std <= 0:
            raise ValueError("init_std must be > 0")

    @property
    def query_width(self) -> int:
        return self.num_heads * self.q_head_dim

    @property
    def value_width(self) -> int:
        return self.num_heads * self.v_head_dim

    @property
    def full_kv_width(self) -> int:
        return self.num_heads * (self.q_head_dim + self.v_head_dim)

    @property
    def cache_compression_ratio(self) -> float:
        return self.full_kv_width / self.kv_latent_dim

    @property
    def resolved_backend(self) -> Literal["manual", "sdpa"]:
        if self.attention_backend == "auto":
            return "sdpa" if self.use_sdpa else "manual"
        return self.attention_backend

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "GatedMLAConfig":
        return cls(**values)
