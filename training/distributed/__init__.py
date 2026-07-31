"""Canonical PyTorch-native DP/TP/EP utilities for Kimi K3."""

from .collectives import (
    all_gather_variable,
    all_ranks_true,
    all_reduce_max,
    all_reduce_mean,
    all_reduce_sum,
    any_rank_true,
    group_rank,
    group_size,
)
from .config import (
    DataParallelConfig,
    DistributedCheckpointConfig,
    DistributedConfig,
    DistributedDiagnosticsConfig,
    ExpertParallelConfig,
    FSDPConfig,
    InactiveParallelConfig,
    TensorParallelConfig,
    distributed_config_from_dict,
)
from .environment import (
    DistributedContext,
    DistributedEnvironment,
    initialize_distributed,
)
from .checkpoint import (
    export_rank0_full_checkpoint,
    load_distributed_checkpoint,
    save_distributed_checkpoint,
)
from .expert_parallel import (
    ExpertParallelMoE,
    all_to_all_expert_dispatch,
    shard_kimi_experts,
)
from .kimi_tensor_parallel import (
    TensorParallelKDA,
    TensorParallelMLA,
    shard_kimi_attention,
)
from .mesh import (
    build_device_mesh,
    coordinates_from_rank,
    rank_from_coordinates,
)
from .metrics import (
    print_topology,
    reduce_counter,
    reduce_scalar_metrics,
    reduce_weighted_mean,
    topology_lines,
)
from .sampler import StatefulDistributedSampler
from .parallelize import (
    parallelize_kimi_model,
    validate_distributed_model_config,
)
from .tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelMetadata,
    VocabParallelEmbedding,
    VocabParallelLMHead,
    shard_vocabulary,
    vocab_parallel_cross_entropy,
)
from .wrapping import (
    accumulation_sync_context,
    broadcast_model_state,
    clip_grad_norm,
    distributed_grad_norm,
    unwrap_model,
    wrap_data_parallel,
)

__all__ = [
    "DataParallelConfig",
    "DistributedCheckpointConfig",
    "DistributedConfig",
    "DistributedContext",
    "DistributedDiagnosticsConfig",
    "DistributedEnvironment",
    "ExpertParallelMoE",
    "ExpertParallelConfig",
    "FSDPConfig",
    "InactiveParallelConfig",
    "StatefulDistributedSampler",
    "TensorParallelKDA",
    "TensorParallelMLA",
    "TensorParallelMetadata",
    "TensorParallelConfig",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "VocabParallelLMHead",
    "accumulation_sync_context",
    "all_gather_variable",
    "all_ranks_true",
    "all_reduce_max",
    "all_reduce_mean",
    "all_reduce_sum",
    "any_rank_true",
    "broadcast_model_state",
    "build_device_mesh",
    "clip_grad_norm",
    "distributed_grad_norm",
    "coordinates_from_rank",
    "distributed_config_from_dict",
    "export_rank0_full_checkpoint",
    "group_rank",
    "group_size",
    "initialize_distributed",
    "load_distributed_checkpoint",
    "parallelize_kimi_model",
    "print_topology",
    "rank_from_coordinates",
    "reduce_counter",
    "reduce_scalar_metrics",
    "reduce_weighted_mean",
    "topology_lines",
    "unwrap_model",
    "save_distributed_checkpoint",
    "shard_kimi_attention",
    "shard_kimi_experts",
    "shard_vocabulary",
    "all_to_all_expert_dispatch",
    "validate_distributed_model_config",
    "vocab_parallel_cross_entropy",
    "wrap_data_parallel",
]
