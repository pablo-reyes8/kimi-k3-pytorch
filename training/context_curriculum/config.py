"""Validated configuration for progressive context extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ContextStage:
    max_seq_len: int
    until_tokens: int | None

    def __post_init__(self) -> None:
        if self.max_seq_len <= 0:
            raise ValueError("context stage max_seq_len must be positive")
        if self.until_tokens is not None and self.until_tokens <= 0:
            raise ValueError("context stage until_tokens must be positive")


def _default_stages() -> tuple[ContextStage, ...]:
    return (
        ContextStage(512, 20_000_000),
        ContextStage(1024, 40_000_000),
        ContextStage(2048, 60_000_000),
        ContextStage(4096, 80_000_000),
        ContextStage(8192, None),
    )


@dataclass(frozen=True)
class ContextCurriculumConfig:
    enabled: bool = False
    stages: tuple[ContextStage, ...] = field(default_factory=_default_stages)
    transition_unit: Literal["tokens"] = "tokens"
    reset_dataloader_on_transition: bool = True
    preserve_optimizer_state: bool = True
    preserve_scheduler_state: bool = True

    def __post_init__(self) -> None:
        stages = tuple(
            ContextStage(**stage) if isinstance(stage, dict) else stage
            for stage in self.stages
        )
        object.__setattr__(self, "stages", stages)
        if not stages:
            raise ValueError("context curriculum stages must not be empty")
        lengths = [stage.max_seq_len for stage in stages]
        if any(right <= left for left, right in zip(lengths, lengths[1:])):
            raise ValueError("stage max_seq_len values must strictly increase")
        if stages[-1].until_tokens is not None:
            raise ValueError("the final context stage must be open-ended")
        finite = [stage.until_tokens for stage in stages[:-1]]
        if any(value is None for value in finite):
            raise ValueError("only the final stage may be open-ended")
        if any(right <= left for left, right in zip(finite, finite[1:])):
            raise ValueError("until_tokens values must strictly increase")
        if self.transition_unit != "tokens":
            raise ValueError("tokens are the only supported transition unit")
        if not self.preserve_optimizer_state:
            raise ValueError("PCC cannot reset optimizer state")
        if not self.preserve_scheduler_state:
            raise ValueError("PCC cannot reset scheduler state")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ContextCurriculumConfig":
        return cls(**values)


__all__ = ["ContextCurriculumConfig", "ContextStage"]
