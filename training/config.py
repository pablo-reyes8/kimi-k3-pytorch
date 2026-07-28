"""Validated configuration objects for the Kimi K3 training stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


def _positive_optional(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be None or a positive integer")


@dataclass(frozen=True)
class TrainingConfig:
    """Core loop configuration independent from model construction."""

    epochs: int = 1
    gradient_accumulation_steps: int = 1
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    grad_clip_norm: float | None = 1.0
    use_mtp: bool = True
    max_batches_per_epoch: int | None = None
    max_eval_batches: int | None = None
    log_every_steps: int = 10
    eval_every_epochs: int = 1
    checkpoint_every_epochs: int = 1
    prediction_every_epochs: int | None = 1
    seed: int = 1337
    deterministic: bool = False
    max_seq_len: int = 8192

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be 'fp32', 'bf16', or 'fp16'")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be None or positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        for name in (
            "max_batches_per_epoch",
            "max_eval_batches",
            "log_every_steps",
            "eval_every_epochs",
            "checkpoint_every_epochs",
            "prediction_every_epochs",
        ):
            _positive_optional(name, getattr(self, name))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PretrainingLossConfig:
    """NTP/MTP objective knobs controlled by the training run."""

    ignore_index: int = -100
    label_smoothing: float = 0.0
    mtp_loss_weight: float | None = None

    def __post_init__(self) -> None:
        if self.ignore_index >= 0:
            raise ValueError("ignore_index must be negative")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.mtp_loss_weight is not None and self.mtp_loss_weight < 0:
            raise ValueError("mtp_loss_weight must be None or non-negative")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW baseline required by the basic Kimi training phase."""

    name: Literal["adamw"] = "adamw"
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ValueError("the basic Kimi engine supports only AdamW")
        if self.learning_rate <= 0 or self.eps <= 0:
            raise ValueError("learning_rate and eps must be positive")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SchedulerConfig:
    """Linear warmup followed by cosine decay."""

    kind: Literal["cosine"] = "cosine"
    warmup_ratio: float | None = 0.01
    warmup_steps: int | None = None
    min_lr_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.kind != "cosine":
            raise ValueError("only the cosine scheduler is supported")
        if self.warmup_ratio is not None and self.warmup_steps is not None:
            raise ValueError("set warmup_ratio or warmup_steps, not both")
        if self.warmup_ratio is not None and not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")

    def resolve_warmup_steps(self, total_steps: int) -> int:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_steps is not None:
            return min(self.warmup_steps, total_steps)
        resolved = int(total_steps * (self.warmup_ratio or 0.0))
        if (self.warmup_ratio or 0.0) > 0 and total_steps > 1:
            resolved = max(1, resolved)
        return min(resolved, total_steps)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointConfig:
    """Epoch-boundary checkpoint policy for the high-level orchestrator."""

    output_dir: str | Path = "checkpoints/kimi_k3_mini"
    run_name: str = "kimi_k3_mini"
    keep_last_n: int = 3
    save_optimizer: bool = True
    save_rng_state: bool = True
    atomic_write: bool = True
    resume_from: str | Path | None = None

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run_name must not be empty")
        if self.keep_last_n <= 0:
            raise ValueError("keep_last_n must be positive")
        if not self.atomic_write:
            raise ValueError("atomic checkpoint writes cannot be disabled")

    def to_dict(self) -> dict:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        values["resume_from"] = (
            None if self.resume_from is None else str(self.resume_from)
        )
        return values


@dataclass(frozen=True)
class PredictionConfig:
    """Small qualitative next-token preview policy."""

    sample_index: int = 0
    max_tokens: int = 32

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
