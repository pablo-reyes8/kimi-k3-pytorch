"""Atomic same-topology distributed checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from contextlib import nullcontext

from .environment import DistributedContext
from .wrapping import unwrap_model


FORMAT_VERSION = 1
SUCCESS_MARKER = "SUCCESS"


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, default=str, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _metadata(
    context: DistributedContext,
    *,
    step: int,
    model_config=None,
    data_config=None,
    training_config=None,
    registry_fingerprint: str | None = None,
) -> dict[str, Any]:
    topology = {
        "world_size": context.world_size,
        "dp_size": context.dp_size,
        "tp_size": context.tp_size,
        "ep_size": context.ep_size,
        "backend": context.backend,
    }
    return {
        "format_version": FORMAT_VERSION,
        "step": int(step),
        "topology": topology,
        "model_fingerprint": _fingerprint(model_config),
        "data_fingerprint": _fingerprint(data_config),
        "training_fingerprint": _fingerprint(training_config),
        "parameter_registry_fingerprint": registry_fingerprint,
        "cache_schema_version": 1,
    }


def _barrier(context: DistributedContext) -> None:
    if context.initialized and context.world_size > 1:
        dist.barrier()


def _fsdp_state_context(model: nn.Module):
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel,
            ShardedStateDictConfig,
            StateDictType,
        )
    except ImportError:
        return nullcontext(), False
    if not isinstance(model, FullyShardedDataParallel):
        return nullcontext(), False
    return (
        FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ),
        True,
    )


def save_distributed_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    context: DistributedContext,
    step: int,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    trainer_state=None,
    curriculum=None,
    diagnostics=None,
    sampler=None,
    model_config=None,
    data_config=None,
    training_config=None,
    registry_fingerprint: str | None = None,
    extra: dict[str, Any] | None = None,
    save_rng: bool = True,
) -> Path:
    """Commit rank shards only after every writer reaches the barrier."""
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    if context.is_global_zero:
        if target.exists():
            raise FileExistsError(f"checkpoint already exists: {target}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
    _barrier(context)
    state_context, is_fsdp = _fsdp_state_context(model)
    with state_context:
        model_state = (
            model.state_dict()
            if is_fsdp
            else unwrap_model(model).state_dict()
        )
    optimizer_state = None
    if optimizer is not None:
        if is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel

            optimizer_state = FullyShardedDataParallel.optim_state_dict(
                model, optimizer
            )
        else:
            optimizer_state = optimizer.state_dict()
    payload = {
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "trainer_state": (
            None if trainer_state is None else trainer_state.state_dict()
        ),
        "curriculum": (
            None if curriculum is None else curriculum.state_dict()
        ),
        "diagnostics": (
            None if diagnostics is None else diagnostics.state_dict()
        ),
        "sampler": None if sampler is None else sampler.state_dict(),
        "rng": _rng_state() if save_rng else None,
        "extra": extra or {},
    }
    torch.save(
        payload,
        temporary / f"rank_{context.global_rank:05d}.pt",
    )
    _barrier(context)
    if context.is_global_zero:
        metadata = _metadata(
            context,
            step=step,
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            registry_fingerprint=registry_fingerprint,
        )
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / SUCCESS_MARKER).write_text("ok\n", encoding="utf-8")
        os.replace(temporary, target)
    _barrier(context)
    return target


def load_distributed_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    context: DistributedContext,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    trainer_state=None,
    curriculum=None,
    diagnostics=None,
    sampler=None,
    restore_rng: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """Restore the exact rank shard and reject topology changes."""
    source = Path(path)
    if not (source / SUCCESS_MARKER).is_file():
        raise ValueError(
            f"{source} is incomplete: missing {SUCCESS_MARKER} marker"
        )
    metadata = json.loads(
        (source / "metadata.json").read_text(encoding="utf-8")
    )
    expected = {
        "world_size": context.world_size,
        "dp_size": context.dp_size,
        "tp_size": context.tp_size,
        "ep_size": context.ep_size,
    }
    actual = {
        key: metadata["topology"][key] for key in expected
    }
    if actual != expected:
        raise ValueError(
            "distributed checkpoint topology change is unsupported: "
            f"saved={actual}, current={expected}"
        )
    payload = torch.load(
        source / f"rank_{context.global_rank:05d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    state_context, is_fsdp = _fsdp_state_context(model)
    with state_context:
        incompatible = (
            model.load_state_dict(payload["model"], strict=strict)
            if is_fsdp
            else unwrap_model(model).load_state_dict(
                payload["model"], strict=strict
            )
        )
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer_state = payload["optimizer"]
        if is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel

            optimizer_state = (
                FullyShardedDataParallel.optim_state_dict_to_load(
                    model, optimizer, optimizer_state
                )
            )
        optimizer.load_state_dict(optimizer_state)
    for owner, key in (
        (scheduler, "scheduler"),
        (scaler, "scaler"),
        (ema, "ema"),
        (trainer_state, "trainer_state"),
        (curriculum, "curriculum"),
        (diagnostics, "diagnostics"),
        (sampler, "sampler"),
    ):
        if owner is not None and payload.get(key) is not None:
            owner.load_state_dict(payload[key])
    if restore_rng:
        _restore_rng_state(payload.get("rng"))
    return {
        "metadata": metadata,
        "extra": payload.get("extra", {}),
        "sampler_state": payload.get("sampler"),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def export_rank0_full_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    context: DistributedContext,
) -> Path | None:
    """Export replicated DDP/single models; TP/EP exports need resharding."""
    if context.tp_size != 1 or context.ep_size != 1:
        raise RuntimeError(
            "rank-0 full export from TP/EP requires explicit resharding and "
            "is not claimed by this phase"
        )
    if not context.is_global_zero:
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": unwrap_model(model).state_dict()}, target)
    return target


__all__ = [
    "FORMAT_VERSION",
    "SUCCESS_MARKER",
    "export_rank0_full_checkpoint",
    "load_distributed_checkpoint",
    "save_distributed_checkpoint",
]
