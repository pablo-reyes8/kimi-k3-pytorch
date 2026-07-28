"""Serializable state owned by the Kimi training orchestrator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class TrainerState:
    epoch: int = 0
    optimizer_step: int = 0
    micro_step: int = 0
    samples_seen: int = 0
    tokens_seen: int = 0
    valid_ntp_tokens_seen: int = 0
    valid_mtp_tokens_seen: int = 0
    curriculum_stage_index: int = 0
    best_eval_loss: float | None = None
    skipped_optimizer_steps: int = 0

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, values: dict) -> None:
        known = {field.name for field in fields(self)}
        for name, value in values.items():
            if name in known:
                setattr(self, name, value)

    @classmethod
    def from_state_dict(cls, values: dict | None) -> "TrainerState":
        state = cls()
        if values:
            state.load_state_dict(values)
        return state
