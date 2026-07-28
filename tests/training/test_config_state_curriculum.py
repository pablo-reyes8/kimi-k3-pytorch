from dataclasses import replace

import pytest

from training import (
    CheckpointConfig,
    ContextCurriculum,
    ContextCurriculumConfig,
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
        ContextCurriculumConfig(
            enabled=True,
            stages=(
                ContextStage(8, 2),
                ContextStage(16, 5),
                ContextStage(32, None),
            ),
        ),
        training_max_seq_len=32,
        model_max_seq_len=32,
    )
    assert not curriculum.update(0)
    assert curriculum.current_max_seq_len() == 8
    assert curriculum.update(2)
    assert curriculum.current_max_seq_len() == 16
    curriculum.validate_sequence_length(16)
    with pytest.raises(ValueError, match="exceeds"):
        curriculum.validate_sequence_length(17)

    state = curriculum.state_dict()
    restored = ContextCurriculum(
        curriculum.config,
        training_max_seq_len=32,
        model_max_seq_len=32,
    )
    restored.load_state_dict(state)
    assert restored.stage_index == 1


@pytest.mark.parametrize(
    "stages",
    [
        [ContextStage(8, 1), ContextStage(8, None)],
        [ContextStage(8, 4), ContextStage(16, 2), ContextStage(32, None)],
        [ContextStage(16, 2), ContextStage(8, None)],
    ],
)
def test_context_curriculum_rejects_invalid_stage_layout(stages):
    with pytest.raises(ValueError):
        ContextCurriculumConfig(enabled=True, stages=tuple(stages))
