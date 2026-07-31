"""Validate and apply Kimi's composable TP/EP model transformation."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from configuration.yaml_utils import ConfigError

from .collectives import group_size
from .config import DistributedConfig
from .environment import DistributedContext
from .expert_parallel import (
    local_expert_parameter_ids,
    scale_local_expert_gradients,
    shard_kimi_experts,
)
from .kimi_tensor_parallel import shard_kimi_attention
from .tensor_parallel import shard_vocabulary


def validate_distributed_model_config(
    config: DistributedConfig, model_config
) -> None:
    """Reject invalid topologies before model or optimizer allocation."""
    if not config.enabled:
        return
    tp = config.tensor_parallel
    ep = config.expert_parallel
    backbone = model_config.backbone
    kda = backbone.kda_config
    mla = backbone.mla_config
    moe = backbone.stable_latent_moe_config
    if config.checkpoint.format != "distributed":
        raise ConfigError(
            "distributed checkpoint format rank0_full is export-only; "
            "training must use format=distributed and call "
            "export_rank0_full_checkpoint explicitly"
        )
    if (
        config.data_parallel.mode == "fsdp"
        and config.fsdp.state_dict_type != "sharded"
    ):
        raise ConfigError(
            "FSDP training checkpoints currently require "
            "fsdp.state_dict_type=sharded"
        )
    if (
        config.data_parallel.mode == "ddp"
        and not config.data_parallel.find_unused_parameters
        and moe.num_routed_experts > moe.top_k
    ):
        raise ConfigError(
            "DDP over sparse routed experts requires "
            "data_parallel.find_unused_parameters=true"
        )
    if tp.enabled:
        if tp.sequence_parallel_norms:
            raise ConfigError(
                "sequence_parallel_norms is reserved but not implemented; "
                "use false"
            )
        divisibility = {
            "model d_model": model_config.d_model,
            "KDA heads": kda.num_heads,
            "MLA heads": mla.num_heads,
        }
        if tp.shard_embeddings:
            divisibility["vocabulary"] = model_config.vocab_size
        if tp.shard_moe_latent_projections:
            divisibility["MoE latent_dim"] = moe.latent_dim
        invalid = {
            name: value
            for name, value in divisibility.items()
            if value % tp.size
        }
        if invalid:
            details = ", ".join(
                f"{name}={value}" for name, value in invalid.items()
            )
            raise ConfigError(
                f"TP size {tp.size} does not divide {details}"
            )
        if not tp.shard_kda_heads or not tp.shard_mla_heads:
            raise ConfigError(
                "the Kimi TP plan requires both KDA and MLA head sharding"
            )
        if tp.shard_moe_latent_projections:
            raise ConfigError(
                "shard_moe_latent_projections is not part of the current "
                "complete-head baseline; use EP to shard routed experts"
            )
    if ep.enabled:
        if moe.num_routed_experts % ep.size:
            raise ConfigError(
                f"routed experts={moe.num_routed_experts} must be divisible "
                f"by EP size={ep.size}"
            )
        if moe.top_k > moe.num_routed_experts:
            raise ConfigError("MoE top-k exceeds routed expert count")
    if (
        config.data_parallel.mode == "fsdp"
        and config.fsdp.state_dict_type != config.checkpoint.format
        and config.checkpoint.format != "distributed"
    ):
        raise ConfigError("FSDP and distributed checkpoint formats disagree")


def _register_ep_replica_gradient_hooks(
    model: nn.Module, group
) -> None:
    excluded = local_expert_parameter_ids(model)
    for parameter in model.parameters():
        if id(parameter) in excluded or not parameter.requires_grad:
            continue

        def synchronize(gradient, process_group=group):
            if group_size(process_group) > 1:
                dist.all_reduce(gradient, group=process_group)
                gradient.div_(group_size(process_group))
            return gradient

        parameter.register_hook(synchronize)


def _mark_parallel_parameter_ownership(model: nn.Module) -> None:
    from .expert_parallel import ExpertParallelMoE
    from .kimi_tensor_parallel import TensorParallelKDA, TensorParallelMLA
    from .tensor_parallel import (
        ColumnParallelLinear,
        RowParallelLinear,
        VocabParallelEmbedding,
        VocabParallelLMHead,
    )

    for parameter in model.parameters():
        parameter._kimi_tp_sharded = False
        parameter._kimi_ep_sharded = False
    for module in model.modules():
        if isinstance(module, ColumnParallelLinear):
            for parameter in module.parameters(recurse=False):
                parameter._kimi_tp_sharded = True
        elif isinstance(module, RowParallelLinear):
            module.weight._kimi_tp_sharded = True
        elif isinstance(module, (VocabParallelEmbedding, VocabParallelLMHead)):
            module.weight._kimi_tp_sharded = True
            if getattr(module, "bias", None) is not None:
                module.bias._kimi_tp_sharded = True
        elif isinstance(module, ExpertParallelMoE):
            for parameter in module.routed_experts.parameters():
                parameter._kimi_ep_sharded = True
        elif isinstance(module, (TensorParallelKDA, TensorParallelMLA)):
            for parameter in module.parameters():
                parameter._kimi_tp_sharded = True
            for parameter in module.compression.parameters() if isinstance(
                module, TensorParallelMLA
            ) else module.alpha_down.parameters():
                parameter._kimi_tp_sharded = False
            if module.output_proj.bias is not None:
                module.output_proj.bias._kimi_tp_sharded = False


def parallelize_kimi_model(
    model: nn.Module,
    config: DistributedConfig,
    context: DistributedContext,
) -> dict[str, object]:
    """Transform the one existing Kimi model in place."""
    validate_distributed_model_config(config, model.config)
    report: dict[str, object] = {
        "tensor_parallel": False,
        "expert_parallel": False,
        "attention_layers_sharded": 0,
        "moe_layers_sharded": 0,
        "mla_cache_layout": "replicated",
    }
    if config.tensor_parallel.enabled:
        attention = shard_kimi_attention(model, group=context.tp_group)
        if config.tensor_parallel.shard_embeddings:
            vocabulary = shard_vocabulary(
                model, group=context.tp_group, gather_logits=True
            )
            report["vocab_range"] = (
                vocabulary.vocab_start,
                vocabulary.vocab_end,
            )
        report.update(
            tensor_parallel=True,
            attention_layers_sharded=(
                attention.transformed_attention_layers
            ),
        )
    if config.expert_parallel.enabled:
        count = shard_kimi_experts(model, group=context.ep_group)
        _register_ep_replica_gradient_hooks(model, context.ep_group)
        scale_local_expert_gradients(model, group=context.ep_group)
        report.update(
            expert_parallel=True,
            moe_layers_sharded=count,
        )
    _mark_parallel_parameter_ownership(model)
    return report


__all__ = [
    "parallelize_kimi_model",
    "validate_distributed_model_config",
]
