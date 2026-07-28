"""Portable, atomic training checkpoints."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def _rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _config_dict(config: Any) -> Any:
    if config is None or isinstance(config, (str, int, float, bool, list, dict)):
        return config
    if is_dataclass(config):
        return asdict(config)
    return dict(vars(config)) if hasattr(config, "__dict__") else str(config)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch: int = 0,
    global_step: int = 0,
    model_config=None,
    training_config=None,
    history=None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_config": _config_dict(model_config),
        "training_config": _config_dict(training_config),
        "history": history or {},
        "rng_state": _rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = False,
) -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    incompatible = raw_model.load_state_dict(payload["model_state_dict"], strict=strict)
    for obj, key in (
        (optimizer, "optimizer_state_dict"),
        (scheduler, "scheduler_state_dict"),
        (scaler, "scaler_state_dict"),
    ):
        if obj is not None and payload.get(key) is not None:
            obj.load_state_dict(payload[key])
    if restore_rng:
        _restore_rng(payload.get("rng_state"))
    return {
        "epoch": payload.get("epoch", 0),
        "global_step": payload.get("global_step", 0),
        "model_config": payload.get("model_config"),
        "training_config": payload.get("training_config"),
        "history": payload.get("history", {}),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
