"""Token-based progressive context state machine."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import ContextCurriculumConfig


@dataclass
class ContextCurriculumState:
    stage_index: int
    max_seq_len: int
    tokens_seen: int
    transition_count: int = 0


@dataclass(frozen=True)
class ContextTransition:
    old_stage_index: int
    new_stage_index: int
    old_max_seq_len: int
    new_max_seq_len: int
    tokens_seen_at_transition: int

    def to_dict(self) -> dict:
        return asdict(self)


class ProgressiveContextCurriculum:
    """Determine the active context without mutating training components."""

    def __init__(
        self,
        config: ContextCurriculumConfig,
        *,
        training_max_seq_len: int,
        model_max_seq_len: int | None = None,
        mtp_min_seq_len: int = 1,
    ):
        if training_max_seq_len <= 0:
            raise ValueError("training_max_seq_len must be positive")
        if model_max_seq_len is not None and model_max_seq_len <= 0:
            raise ValueError("model_max_seq_len must be positive")
        if mtp_min_seq_len <= 0:
            raise ValueError("mtp_min_seq_len must be positive")
        self.config = config
        self.training_max_seq_len = int(training_max_seq_len)
        self.model_max_seq_len = (
            None if model_max_seq_len is None else int(model_max_seq_len)
        )
        self.mtp_min_seq_len = int(mtp_min_seq_len)
        limit = self.training_max_seq_len
        if self.model_max_seq_len is not None:
            limit = min(limit, self.model_max_seq_len)
        if config.enabled:
            if config.stages[-1].max_seq_len > limit:
                raise ValueError(
                    "context stage exceeds the model/training max_seq_len"
                )
            if config.stages[0].max_seq_len < mtp_min_seq_len:
                raise ValueError(
                    "first context stage is too short for enabled MTP"
                )
            initial_length = config.stages[0].max_seq_len
        else:
            initial_length = limit
        self.state = ContextCurriculumState(0, initial_length, 0, 0)
        self.last_transition: ContextTransition | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def stage_index(self) -> int:
        return self.state.stage_index

    def current_max_seq_len(self) -> int:
        return self.state.max_seq_len

    def validate_sequence_length(self, sequence_length: int) -> None:
        if sequence_length > self.current_max_seq_len():
            raise ValueError(
                f"batch sequence length {sequence_length} exceeds active "
                f"context length {self.current_max_seq_len()}"
            )

    def validate_valid_tokens(self, valid_tokens_per_sample) -> None:
        """Reject samples that cannot form the configured MTP target."""
        if bool((valid_tokens_per_sample < self.mtp_min_seq_len).any()):
            raise ValueError(
                "a context-curriculum sample has too few valid tokens for MTP"
            )

    def update(self, tokens_seen: int) -> bool:
        if tokens_seen < self.state.tokens_seen:
            raise ValueError("context curriculum tokens_seen cannot move backward")
        self.state.tokens_seen = int(tokens_seen)
        self.last_transition = None
        if not self.enabled:
            return False
        old_index = self.state.stage_index
        new_index = old_index
        stages = self.config.stages
        while (
            new_index < len(stages) - 1
            and stages[new_index].until_tokens is not None
            and tokens_seen >= stages[new_index].until_tokens
        ):
            new_index += 1
        if new_index == old_index:
            return False
        old_length = self.state.max_seq_len
        self.state.stage_index = new_index
        self.state.max_seq_len = stages[new_index].max_seq_len
        self.state.transition_count += 1
        self.last_transition = ContextTransition(
            old_stage_index=old_index,
            new_stage_index=new_index,
            old_max_seq_len=old_length,
            new_max_seq_len=self.state.max_seq_len,
            tokens_seen_at_transition=int(tokens_seen),
        )
        return True

    def metrics(self) -> dict[str, float]:
        return {
            "context/stage_index": float(self.state.stage_index),
            "context/max_seq_len": float(self.state.max_seq_len),
            "context/tokens_seen": float(self.state.tokens_seen),
            "context/transition_count": float(self.state.transition_count),
        }

    def state_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "stage_index": self.state.stage_index,
            "max_seq_len": self.state.max_seq_len,
            "tokens_seen": self.state.tokens_seen,
            "transition_count": self.state.transition_count,
            "stages": [asdict(stage) for stage in self.config.stages],
            "transition_unit": self.config.transition_unit,
            "reset_dataloader_on_transition": (
                self.config.reset_dataloader_on_transition
            ),
            "preserve_optimizer_state": self.config.preserve_optimizer_state,
            "preserve_scheduler_state": self.config.preserve_scheduler_state,
            "training_max_seq_len": self.training_max_seq_len,
            "model_max_seq_len": self.model_max_seq_len,
            "mtp_min_seq_len": self.mtp_min_seq_len,
        }

    def load_state_dict(self, values: dict) -> None:
        expected_stages = [asdict(stage) for stage in self.config.stages]
        if values.get("stages") != expected_stages:
            raise ValueError("checkpoint context stage definition is incompatible")
        compatibility = {
            "enabled": self.enabled,
            "transition_unit": self.config.transition_unit,
            "reset_dataloader_on_transition": (
                self.config.reset_dataloader_on_transition
            ),
            "preserve_optimizer_state": self.config.preserve_optimizer_state,
            "preserve_scheduler_state": self.config.preserve_scheduler_state,
            "training_max_seq_len": self.training_max_seq_len,
            "model_max_seq_len": self.model_max_seq_len,
            "mtp_min_seq_len": self.mtp_min_seq_len,
        }
        for name, expected in compatibility.items():
            if values.get(name) != expected:
                raise ValueError(
                    f"checkpoint context curriculum {name} is incompatible"
                )
        index = int(values["stage_index"])
        if not 0 <= index < len(self.config.stages):
            raise ValueError("checkpoint context stage index is invalid")
        if index < self.state.stage_index:
            raise ValueError("checkpoint would move context curriculum backward")
        expected_length = (
            self.config.stages[index].max_seq_len
            if self.enabled
            else min(
                self.training_max_seq_len,
                self.model_max_seq_len or self.training_max_seq_len,
            )
        )
        if int(values["max_seq_len"]) != expected_length:
            raise ValueError("checkpoint active context length is inconsistent")
        tokens_seen = int(values["tokens_seen"])
        if tokens_seen < 0:
            raise ValueError("checkpoint curriculum tokens_seen is invalid")
        expected_index = 0
        if self.enabled:
            for stage_index, stage in enumerate(self.config.stages[:-1]):
                if tokens_seen >= stage.until_tokens:
                    expected_index = stage_index + 1
        if index != expected_index:
            raise ValueError("checkpoint stage does not match its token count")
        self.state = ContextCurriculumState(
            stage_index=index,
            max_seq_len=expected_length,
            tokens_seen=tokens_seen,
            transition_count=int(values.get("transition_count", index)),
        )
        self.last_transition = None


__all__ = [
    "ContextCurriculumState",
    "ContextTransition",
    "ProgressiveContextCurriculum",
]
