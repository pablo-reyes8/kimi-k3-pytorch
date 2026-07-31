"""Strict configuration for the PyTorch-native distributed baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from configuration.yaml_utils import ConfigError, dataclass_kwargs


@dataclass(frozen=True)
class DataParallelConfig:
    mode: Literal["none", "ddp", "fsdp"] = "none"
    size: int = 1
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"none", "ddp", "fsdp"}:
            raise ValueError("data_parallel.mode must be none, ddp or fsdp")
        if self.size <= 0:
            raise ValueError("data_parallel.size must be positive")
        if self.mode == "none" and self.size != 1:
            raise ValueError("data_parallel.size must be 1 when mode=none")
        if self.mode != "none" and self.size == 1:
            raise ValueError(
                "data_parallel.size must be > 1 when data parallelism is enabled"
            )


@dataclass(frozen=True)
class FSDPConfig:
    sharding_strategy: Literal[
        "full_shard", "shard_grad_op", "no_shard"
    ] = "full_shard"
    auto_wrap_policy: Literal["kimi_block", "size_based", "none"] = "kimi_block"
    min_num_params: int = 0
    mixed_precision: Literal["inherit", "fp32", "bf16", "fp16"] = "inherit"
    cpu_offload: bool = False
    use_orig_params: bool = True
    forward_prefetch: bool = False
    backward_prefetch: Literal[
        "backward_pre", "backward_post", "none"
    ] = "backward_pre"
    limit_all_gathers: bool = True
    state_dict_type: Literal["sharded", "rank0_full"] = "sharded"

    def __post_init__(self) -> None:
        if self.min_num_params < 0:
            raise ValueError("fsdp.min_num_params must be non-negative")


@dataclass(frozen=True)
class TensorParallelConfig:
    enabled: bool = False
    size: int = 1
    shard_embeddings: bool = True
    shard_lm_head: bool = True
    sequence_parallel_norms: bool = False
    shard_kda_heads: bool = True
    shard_mla_heads: bool = True
    shard_moe_latent_projections: bool = False
    mla_cache_layout: Literal["replicated", "latent_sharded"] = "replicated"

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("tensor_parallel.size must be positive")
        if self.enabled != (self.size > 1):
            raise ValueError(
                "tensor_parallel.enabled must be true exactly when size > 1"
            )
        if self.mla_cache_layout == "latent_sharded":
            raise ValueError(
                "latent-sharded MLA cache is reserved for a later phase; "
                "use mla_cache_layout=replicated"
            )
        if self.shard_embeddings != self.shard_lm_head:
            raise ValueError(
                "tied Kimi vocabulary weights require embedding and LM-head "
                "sharding to be enabled or disabled together"
            )


@dataclass(frozen=True)
class ExpertParallelConfig:
    enabled: bool = False
    size: int = 1
    dispatch: Literal["all_to_all"] = "all_to_all"
    expert_placement: Literal["contiguous"] = "contiguous"
    shared_experts: Literal["replicated"] = "replicated"
    drop_tokens: bool = False
    capacity_factor: float | None = None

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("expert_parallel.size must be positive")
        if self.enabled != (self.size > 1):
            raise ValueError(
                "expert_parallel.enabled must be true exactly when size > 1"
            )
        if self.drop_tokens:
            raise ValueError("Kimi expert parallelism never drops routed tokens")
        if self.capacity_factor is not None:
            raise ValueError(
                "capacity_factor is unsupported by the no-drop baseline"
            )


@dataclass(frozen=True)
class InactiveParallelConfig:
    enabled: bool = False
    size: int = 1

    def __post_init__(self) -> None:
        if self.enabled or self.size != 1:
            raise ValueError(
                "pipeline/context parallelism is validation metadata only "
                "and must remain disabled with size=1"
            )


@dataclass(frozen=True)
class DistributedCheckpointConfig:
    format: Literal["distributed", "rank0_full"] = "distributed"
    async_save: bool = False
    save_rng_per_rank: bool = True

    def __post_init__(self) -> None:
        if self.async_save:
            raise ValueError("asynchronous distributed saves are not implemented")


@dataclass(frozen=True)
class DistributedDiagnosticsConfig:
    aggregate_across_data_parallel: bool = True
    log_rank: int = 0
    per_rank_debug: bool = False

    def __post_init__(self) -> None:
        if self.log_rank < 0:
            raise ValueError("distributed.diagnostics.log_rank must be >= 0")


@dataclass(frozen=True)
class DistributedConfig:
    enabled: bool = False
    backend: Literal["auto", "gloo", "nccl"] = "auto"
    init_method: Literal["env"] = "env"
    timeout_seconds: int = 1800
    data_parallel: DataParallelConfig = field(
        default_factory=DataParallelConfig
    )
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    tensor_parallel: TensorParallelConfig = field(
        default_factory=TensorParallelConfig
    )
    expert_parallel: ExpertParallelConfig = field(
        default_factory=ExpertParallelConfig
    )
    pipeline_parallel: InactiveParallelConfig = field(
        default_factory=InactiveParallelConfig
    )
    context_parallel: InactiveParallelConfig = field(
        default_factory=InactiveParallelConfig
    )
    checkpoint: DistributedCheckpointConfig = field(
        default_factory=DistributedCheckpointConfig
    )
    diagnostics: DistributedDiagnosticsConfig = field(
        default_factory=DistributedDiagnosticsConfig
    )

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("distributed.timeout_seconds must be positive")
        if not self.enabled and self.logical_world_size != 1:
            raise ValueError(
                "distributed.enabled=false requires DP=TP=EP=1"
            )
        if self.enabled and self.logical_world_size <= 1:
            raise ValueError(
                "distributed.enabled=true requires at least one parallel size > 1"
            )
        if self.diagnostics.log_rank >= self.logical_world_size:
            raise ValueError(
                "distributed.diagnostics.log_rank must be inside world size"
            )

    @property
    def logical_world_size(self) -> int:
        return (
            self.data_parallel.size
            * self.tensor_parallel.size
            * self.expert_parallel.size
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nested(values: dict[str, Any], name: str, cls):
    raw = values.pop(name, {})
    if not isinstance(raw, dict):
        raise ConfigError(f"distributed.{name} must be a mapping")
    return cls(
        **dataclass_kwargs(raw, cls, path=f"distributed.{name}")
    )


def distributed_config_from_dict(
    values: dict[str, Any] | None,
) -> DistributedConfig:
    """Build a strict nested configuration from a YAML mapping."""
    root = dict(values or {})
    nested = {
        "data_parallel": _nested(root, "data_parallel", DataParallelConfig),
        "fsdp": _nested(root, "fsdp", FSDPConfig),
        "tensor_parallel": _nested(
            root, "tensor_parallel", TensorParallelConfig
        ),
        "expert_parallel": _nested(
            root, "expert_parallel", ExpertParallelConfig
        ),
        "pipeline_parallel": _nested(
            root, "pipeline_parallel", InactiveParallelConfig
        ),
        "context_parallel": _nested(
            root, "context_parallel", InactiveParallelConfig
        ),
        "checkpoint": _nested(
            root, "checkpoint", DistributedCheckpointConfig
        ),
        "diagnostics": _nested(
            root, "diagnostics", DistributedDiagnosticsConfig
        ),
    }
    return DistributedConfig(
        **dataclass_kwargs(root, DistributedConfig, path="distributed"),
        **nested,
    )


__all__ = [
    "DataParallelConfig",
    "DistributedCheckpointConfig",
    "DistributedConfig",
    "DistributedDiagnosticsConfig",
    "ExpertParallelConfig",
    "FSDPConfig",
    "InactiveParallelConfig",
    "TensorParallelConfig",
    "distributed_config_from_dict",
]
