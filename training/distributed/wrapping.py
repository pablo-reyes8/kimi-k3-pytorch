"""DDP/FSDP wrapping at the boundary before optimizer construction."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .config import DistributedConfig
from .environment import DistributedContext


def broadcast_model_state(
    model: nn.Module, context: DistributedContext, *, source: int = 0
) -> None:
    """Make pre-wrap initialization identical without serializing a model."""
    if not context.initialized or context.world_size == 1:
        return
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.numel():
            torch.distributed.broadcast(tensor.data, src=source)


def _fsdp_strategy(name: str):
    from torch.distributed.fsdp import ShardingStrategy

    return {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[name]


def _fsdp_backward_prefetch(name: str):
    from torch.distributed.fsdp import BackwardPrefetch

    return {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
        "none": None,
    }[name]


def _fsdp_mixed_precision(name: str, inherited: str):
    from torch.distributed.fsdp import MixedPrecision

    resolved = inherited if name == "inherit" else name
    if resolved == "fp32":
        return None
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[resolved]
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


def wrap_data_parallel(
    model: nn.Module,
    config: DistributedConfig,
    context: DistributedContext,
    *,
    training_precision: str,
) -> nn.Module:
    """Wrap on the DP axis only; TP and EP remain orthogonal."""
    mode = config.data_parallel.mode
    if mode == "none":
        return model
    if context.dp_size <= 1:
        raise RuntimeError(f"{mode.upper()} requires a DP group larger than one")
    if mode == "ddp":
        cuda = context.device.type == "cuda"
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if cuda else None,
            output_device=context.local_rank if cuda else None,
            process_group=context.dp_group,
            find_unused_parameters=(
                config.data_parallel.find_unused_parameters
            ),
            gradient_as_bucket_view=(
                config.data_parallel.gradient_as_bucket_view
            ),
            static_graph=config.data_parallel.static_graph,
        )

    if context.device.type == "cpu":
        raise RuntimeError(
            "this installed PyTorch build requires an accelerator for FSDP; "
            "use DDP for CPU/Gloo validation"
        )
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    fsdp = config.fsdp
    if fsdp.auto_wrap_policy == "size_based":
        auto_wrap = partial(
            size_based_auto_wrap_policy,
            min_num_params=fsdp.min_num_params,
        )
    elif fsdp.auto_wrap_policy == "kimi_block":
        from src.hybrid_backbone.attention_layer import HybridAttentionLayer

        def auto_wrap(module, recurse, nonwrapped_numel):
            if recurse:
                return True
            return isinstance(module, HybridAttentionLayer)
    else:
        auto_wrap = None
    return FullyShardedDataParallel(
        model,
        process_group=context.dp_group,
        sharding_strategy=_fsdp_strategy(fsdp.sharding_strategy),
        cpu_offload=CPUOffload(offload_params=fsdp.cpu_offload),
        auto_wrap_policy=auto_wrap,
        backward_prefetch=_fsdp_backward_prefetch(fsdp.backward_prefetch),
        mixed_precision=_fsdp_mixed_precision(
            fsdp.mixed_precision, training_precision
        ),
        device_id=context.device,
        forward_prefetch=fsdp.forward_prefetch,
        limit_all_gathers=fsdp.limit_all_gathers,
        use_orig_params=fsdp.use_orig_params,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def accumulation_sync_context(model: nn.Module, *, synchronize: bool):
    """Return DDP/FSDP no_sync only for non-final microbatches."""
    if synchronize or not hasattr(model, "no_sync"):
        return nullcontext()
    return model.no_sync()


def distributed_grad_norm(
    model: nn.Module,
    context: DistributedContext | None,
) -> torch.Tensor | None:
    """L2 norm over unique logical parameters across DP×TP×EP."""
    gradients = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return None
    if context is None or not context.initialized:
        squares = torch.stack(
            [parameter.grad.detach().float().square().sum() for parameter in gradients]
        ).sum()
        return squares.sqrt()
    # DDP replicas are identical. TP/EP shards contribute on every owner rank;
    # replicated tensors contribute only at coordinate zero on that axis.
    total = torch.zeros((), device=context.device, dtype=torch.float64)
    for parameter in gradients:
        if context.dp_rank != 0:
            continue
        if (
            context.tp_rank != 0
            and not getattr(parameter, "_kimi_tp_sharded", False)
        ):
            continue
        if (
            context.ep_rank != 0
            and not getattr(parameter, "_kimi_ep_sharded", False)
        ):
            continue
        total += parameter.grad.detach().double().square().sum()
    torch.distributed.all_reduce(total)
    return total.sqrt()


def clip_grad_norm(
    model: nn.Module,
    max_norm: float,
    *,
    context: DistributedContext | None = None,
) -> torch.Tensor:
    """Use FSDP's collective-aware clipping when applicable."""
    if hasattr(model, "clip_grad_norm_"):
        return model.clip_grad_norm_(max_norm)
    norm = distributed_grad_norm(model, context)
    if norm is None:
        return torch.zeros(())
    coefficient = min(1.0, float(max_norm) / (float(norm.item()) + 1e-6))
    if coefficient < 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
    return norm


__all__ = [
    "accumulation_sync_context",
    "broadcast_model_state",
    "clip_grad_norm",
    "distributed_grad_norm",
    "unwrap_model",
    "wrap_data_parallel",
]
