"""Process-group lifecycle and rank-local runtime state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
import os
from typing import Mapping

import torch
import torch.distributed as dist

from .config import DistributedConfig


def _env_int(
    environment: Mapping[str, str], name: str, default: int
) -> int:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class DistributedEnvironment:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int

    @classmethod
    def from_environ(
        cls, environment: Mapping[str, str] | None = None
    ) -> "DistributedEnvironment":
        env = os.environ if environment is None else environment
        world_size = _env_int(env, "WORLD_SIZE", 1)
        if world_size <= 0:
            raise ValueError("WORLD_SIZE must be positive")
        local_world_size = _env_int(env, "LOCAL_WORLD_SIZE", world_size)
        if local_world_size <= 0:
            raise ValueError("LOCAL_WORLD_SIZE must be positive")
        result = cls(
            rank=_env_int(env, "RANK", 0),
            world_size=world_size,
            local_rank=_env_int(env, "LOCAL_RANK", 0),
            local_world_size=local_world_size,
        )
        if result.rank >= result.world_size:
            raise ValueError("RANK must be smaller than WORLD_SIZE")
        if result.local_rank >= result.local_world_size:
            raise ValueError(
                "LOCAL_RANK must be smaller than LOCAL_WORLD_SIZE"
            )
        return result


@dataclass(frozen=True)
class DistributedContext:
    initialized: bool
    owns_process_group: bool
    backend: str
    global_rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: torch.device
    mesh: object | None = None
    dp_group: object | None = None
    tp_group: object | None = None
    ep_group: object | None = None
    dp_rank: int = 0
    tp_rank: int = 0
    ep_rank: int = 0
    dp_size: int = 1
    tp_size: int = 1
    ep_size: int = 1

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def should_log(self) -> bool:
        return self.is_global_zero

    def barrier(self) -> None:
        if self.initialized and self.world_size > 1:
            dist.barrier()

    def close(self) -> None:
        if (
            self.owns_process_group
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.destroy_process_group()


def _resolve_backend(config: DistributedConfig) -> str:
    if config.backend != "auto":
        if config.backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL requires CUDA")
        return config.backend
    return "nccl" if torch.cuda.is_available() else "gloo"


def initialize_distributed(
    config: DistributedConfig,
    *,
    environment: Mapping[str, str] | None = None,
) -> DistributedContext:
    """Initialize an env:// process group once and return its ownership."""
    env = DistributedEnvironment.from_environ(environment)
    if not config.enabled:
        if env.world_size != 1:
            raise RuntimeError(
                "torchrun detected WORLD_SIZE>1 but distributed.enabled=false"
            )
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        return DistributedContext(
            initialized=False,
            owns_process_group=False,
            backend="none",
            global_rank=0,
            local_rank=0,
            world_size=1,
            local_world_size=1,
            device=device,
        )
    if env.world_size != config.logical_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={env.world_size} does not match configured "
            f"DP×TP×EP={config.logical_world_size}"
        )
    backend = _resolve_backend(config)
    device = (
        torch.device("cuda", env.local_rank)
        if backend == "nccl"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=env.rank,
            world_size=env.world_size,
            timeout=timedelta(seconds=config.timeout_seconds),
        )
    elif dist.get_world_size() != env.world_size or dist.get_rank() != env.rank:
        raise RuntimeError("existing process group disagrees with torchrun env")
    return DistributedContext(
        initialized=True,
        owns_process_group=owned,
        backend=backend,
        global_rank=env.rank,
        local_rank=env.local_rank,
        world_size=env.world_size,
        local_world_size=env.local_world_size,
        device=device,
        dp_size=config.data_parallel.size,
        tp_size=config.tensor_parallel.size,
        ep_size=config.expert_parallel.size,
    )


def with_topology(
    context: DistributedContext, **updates
) -> DistributedContext:
    return replace(context, **updates)


__all__ = [
    "DistributedContext",
    "DistributedEnvironment",
    "initialize_distributed",
    "with_topology",
]
