"""Reconstruct a Kimi model and restore training-checkpoint weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src import build_model_from_yaml

from .config import ModelLoadConfig


class _TokenizerModelMetadata:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        value = getattr(self.tokenizer, "vocab_size", None)
        if callable(value):
            value = value()
        if value is None and hasattr(self.tokenizer, "get_vocab_size"):
            value = self.tokenizer.get_vocab_size()
        if value is None:
            raise ValueError("tokenizer does not expose vocabulary size")
        return int(value)

    def token_id(self, token: str) -> int | None:
        from .tokenization import tokenizer_token_id

        return tokenizer_token_id(self.tokenizer, token)


@dataclass
class LoadedKimiCheckpoint:
    model: Any
    checkpoint_path: Path
    format_version: int
    global_step: int
    epoch: int
    metadata: dict[str, Any]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def resolve_load_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA model loading requested but unavailable")
    return device


def _state_dict_from_payload(payload) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    for key in ("model_state_dict", "state_dict", "model"):
        state = payload.get(key)
        if isinstance(state, dict) and state:
            return state
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise ValueError("checkpoint does not contain model weights")


def load_kimi_checkpoint(
    model_config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    tokenizer=None,
    load_config: ModelLoadConfig | None = None,
) -> LoadedKimiCheckpoint:
    """Allocate the YAML architecture, restore weights and enter eval mode."""
    config = load_config or ModelLoadConfig()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    tokenizer_metadata = (
        None if tokenizer is None else _TokenizerModelMetadata(tokenizer)
    )
    model = build_model_from_yaml(
        model_config_path,
        data_bundle=tokenizer_metadata,
    )
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    incompatible = model.load_state_dict(
        _state_dict_from_payload(payload),
        strict=config.strict,
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device = resolve_load_device(config.device)
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(config.precision)
    if dtype is torch.float16 and device.type == "cpu":
        raise RuntimeError("FP16 inference is unsupported on CPU")
    if dtype is None:
        model.to(device=device)
    else:
        model.to(device=device, dtype=dtype)
    model.requires_grad_(False)
    model.eval()
    mapping = payload if isinstance(payload, dict) else {}
    return LoadedKimiCheckpoint(
        model=model,
        checkpoint_path=checkpoint,
        format_version=int(mapping.get("format_version", 0)),
        global_step=int(mapping.get("global_step", 0)),
        epoch=int(mapping.get("epoch", 0)),
        metadata=dict(mapping.get("metadata") or {}),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )


__all__ = [
    "LoadedKimiCheckpoint",
    "load_kimi_checkpoint",
    "resolve_load_device",
]
