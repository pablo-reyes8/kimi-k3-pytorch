"""Progressive context-length policy with explicit step boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ContextStage:
    start_step: int
    context_length: int

    def __post_init__(self) -> None:
        if self.start_step < 0 or self.context_length <= 0:
            raise ValueError("invalid context curriculum stage")


class ContextCurriculum:
    def __init__(self, stages: tuple[ContextStage, ...] | list[ContextStage]):
        self.stages = tuple(stages)
        if not self.stages or self.stages[0].start_step != 0:
            raise ValueError("context curriculum must start at optimizer step zero")
        starts = [stage.start_step for stage in self.stages]
        lengths = [stage.context_length for stage in self.stages]
        if starts != sorted(set(starts)):
            raise ValueError("stage boundaries must be strictly increasing")
        if any(right < left for left, right in zip(lengths, lengths[1:])):
            raise ValueError("context lengths must be non-decreasing")
        self.stage_index = 0

    @property
    def active(self) -> ContextStage:
        return self.stages[self.stage_index]

    def update(self, optimizer_step: int) -> ContextStage:
        self.stage_index = max(
            index
            for index, stage in enumerate(self.stages)
            if stage.start_step <= optimizer_step
        )
        return self.active

    def validate_sequence_length(self, sequence_length: int) -> None:
        if sequence_length > self.active.context_length:
            raise ValueError(
                f"batch sequence length {sequence_length} exceeds active "
                f"context length {self.active.context_length}"
            )

    def state_dict(self) -> dict:
        return {
            "stage_index": self.stage_index,
            "stages": [asdict(stage) for stage in self.stages],
        }

    def load_state_dict(self, values: dict) -> None:
        expected = [asdict(stage) for stage in self.stages]
        if values.get("stages") != expected:
            raise ValueError("checkpoint context curriculum is incompatible")
        index = int(values["stage_index"])
        if not 0 <= index < len(self.stages):
            raise ValueError("invalid checkpoint curriculum stage")
        self.stage_index = index
