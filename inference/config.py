"""Validated configuration for native cached Kimi generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    add_bos_token: bool = True
    skip_special_tokens: bool = True
    seed: int | None = None
    device: str = "auto"
    return_scores: bool = False
    return_cache: bool = True
    max_total_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be None or positive")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if (
            self.repetition_penalty is not None
            and self.repetition_penalty <= 0
        ):
            raise ValueError(
                "repetition_penalty must be None or positive"
            )
        for name in ("eos_token_id", "pad_token_id"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be None or non-negative")
        if not self.device:
            raise ValueError("device must not be empty")
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be None or positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelLoadConfig:
    device: str = "auto"
    precision: Literal["model", "fp32", "bf16", "fp16"] = "model"
    strict: bool = True

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("device must not be empty")
        if self.precision not in {"model", "fp32", "bf16", "fp16"}:
            raise ValueError("unsupported inference precision")


__all__ = ["GenerationConfig", "ModelLoadConfig"]
