"""Stable LatentMoE routing, expert dispatch, and load-balancing components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class StableLatentMoEConfig:
    """Configure Stable LatentMoE experts, routing, and quantile balancing."""

    d_model: int
    latent_dim: int
    num_shared_experts: int = 2
    num_routed_experts: int = 896
    routed_experts_per_token: int = 16
    shared_expert_hidden_dim: int = 3072
    routed_expert_hidden_dim: int = 3072
    beta_gate: float = 4.0
    beta_up: float = 25.0
    norm_eps: float = 1e-6
    router_eps: float = 1e-9
    router_bias: bool = False
    expert_bias: bool = False
    projection_bias: bool = False
    routing_backend: Literal["reference", "vectorized"] = "vectorized"
    quantile_backend: Literal["exact", "histogram"] = "exact"
    enable_quantile_balancing: bool = True
    router_logits_dtype: Literal["input", "float32"] = "float32"
    routing_weights_dtype: Literal["input", "float32"] = "float32"
    routed_accumulation_dtype: Literal["input", "float32"] = "float32"
    histogram_num_bins: int = 256
    histogram_min_margin: float = -2.0
    histogram_max_margin: float = 2.0
    return_router_diagnostics: bool = False
    init_std: float = 0.02

    def __post_init__(self) -> None:
        positive = (
            "d_model",
            "latent_dim",
            "num_routed_experts",
            "routed_experts_per_token",
            "shared_expert_hidden_dim",
            "routed_expert_hidden_dim",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.num_shared_experts < 1:
            raise ValueError("Kimi Stable LatentMoE requires shared experts")
        if self.routed_experts_per_token > self.num_routed_experts:
            raise ValueError(
                "routed_experts_per_token must not exceed num_routed_experts"
            )
        if (
            self.enable_quantile_balancing
            and self.routed_experts_per_token >= self.num_routed_experts
        ):
            raise ValueError("Quantile Balancing requires top_k < num_experts")
        for name in ("beta_gate", "beta_up", "norm_eps", "router_eps", "init_std"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.routing_backend not in ("reference", "vectorized"):
            raise ValueError("unknown routing_backend")
        if self.quantile_backend not in ("exact", "histogram"):
            raise ValueError("unknown quantile_backend")
        for name in (
            "router_logits_dtype",
            "routing_weights_dtype",
            "routed_accumulation_dtype",
        ):
            if getattr(self, name) not in ("input", "float32"):
                raise ValueError(f"unknown dtype policy for {name}")
        if self.histogram_num_bins < 2:
            raise ValueError("histogram_num_bins must be >= 2")
        if self.histogram_max_margin <= self.histogram_min_margin:
            raise ValueError(
                "histogram_max_margin must exceed histogram_min_margin"
            )

    @property
    def top_k(self) -> int:
        return self.routed_experts_per_token

    @classmethod
    def kimi_k3(cls) -> "StableLatentMoEConfig":
        return cls(
            d_model=7168,
            latent_dim=3584,
            num_shared_experts=2,
            num_routed_experts=896,
            routed_experts_per_token=16,
            shared_expert_hidden_dim=3072,
            routed_expert_hidden_dim=3072,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "StableLatentMoEConfig":
        return cls(**values)
