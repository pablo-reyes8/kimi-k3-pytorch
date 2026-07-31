"""Shared torchrun-like process launcher for CPU/Gloo tests."""

from __future__ import annotations

import os
from pathlib import Path
import socket

import torch.multiprocessing as mp


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def configure_rank(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "LOCAL_WORLD_SIZE": str(world_size),
        }
    )


def launch(worker, output: Path, *args, world_size: int = 2) -> None:
    mp.spawn(
        worker,
        args=(world_size, free_port(), str(output), *args),
        nprocs=world_size,
        join=True,
    )
