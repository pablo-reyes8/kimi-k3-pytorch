from dataclasses import replace

import pytest

from training import (
    CheckpointConfig,
    ContextCurriculum,
    ContextStage,
    OptimizerConfig,
    SchedulerConfig,
    TrainerState,
    TrainingConfig,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"gradient_accumulation_steps": 0},
        {"precision": "tf32"},
        {"grad_clip_norm": 0},
        {"prediction_every_epochs": 0},
    ],
)
def test_training_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_optimizer_and_scheduler_validation_and_resolution():
    with pytest.raises(ValueError):
        OptimizerConfig(betas=(0.9, 1.0))
    with pytest.raises(ValueError):
        SchedulerConfig(warmup_ratio=0.1, warmup_steps=2)
    scheduler = SchedulerConfig(warmup_ratio=0.01)
    assert scheduler.resolve_warmup_steps(1000) == 10
    assert replace(scheduler, warmup_ratio=None, warmup_steps=7).resolve_warmup_steps(9) == 7
    with pytest.raises(ValueError, match="atomic"):
        CheckpointConfig(atomic_write=False)


def test_trainer_state_roundtrip_ignores_future_unknown_fields():
    state = TrainerState(optimizer_step=4, tokens_seen=123)
    restored = TrainerState.from_state_dict(
        {**state.state_dict(), "future_field": "compatible"}
    )
    assert restored.optimizer_step == 4
    assert restored.tokens_seen == 123


def test_context_curriculum_boundaries_validation_and_roundtrip():
    curriculum = ContextCurriculum(
        [ContextStage(0, 8), ContextStage(2, 16), ContextStage(5, 32)]
    )
    assert curriculum.update(0).context_length == 8
    assert curriculum.update(2).context_length == 16
    curriculum.validate_sequence_length(16)
    with pytest.raises(ValueError, match="exceeds"):
        curriculum.validate_sequence_length(17)

    state = curriculum.state_dict()
    restored = ContextCurriculum(
        [ContextStage(0, 8), ContextStage(2, 16), ContextStage(5, 32)]
    )
    restored.load_state_dict(state)
    assert restored.stage_index == 1


@pytest.mark.parametrize(
    "stages",
    [
        [ContextStage(1, 8)],
        [ContextStage(0, 8), ContextStage(0, 16)],
        [ContextStage(0, 16), ContextStage(2, 8)],
    ],
)
def test_context_curriculum_rejects_invalid_stage_layout(stages):
    with pytest.raises(ValueError):
        ContextCurriculum(stages)
