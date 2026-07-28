import pytest
import torch

from training import (
    ContextCurriculumConfig,
    ContextStage,
    ProgressiveContextCurriculum,
)


def config(*stages, enabled=True, reset=True):
    return ContextCurriculumConfig(
        enabled=enabled,
        stages=tuple(stages),
        reset_dataloader_on_transition=reset,
    )


def build(curriculum_config, *, training_max=32, model_max=32, mtp_min=3):
    return ProgressiveContextCurriculum(
        curriculum_config,
        training_max_seq_len=training_max,
        model_max_seq_len=model_max,
        mtp_min_seq_len=mtp_min,
    )


def test_curriculum_disabled_uses_training_or_stricter_model_max_seq_len():
    disabled = build(
        config(ContextStage(4, None), enabled=False),
        training_max=16,
        model_max=12,
    )
    assert not disabled.enabled
    assert disabled.current_max_seq_len() == 12
    assert not disabled.update(100)
    assert disabled.stage_index == 0


@pytest.mark.parametrize(
    "stages,match",
    [
        ((ContextStage(8, 10), ContextStage(8, None)), "strictly increase"),
        ((ContextStage(8, 20), ContextStage(16, 10), ContextStage(32, None)),
         "strictly increase"),
        ((ContextStage(8, 10), ContextStage(16, 20)), "open-ended"),
        ((ContextStage(8, None), ContextStage(16, None)), "only the final"),
    ],
)
def test_invalid_stage_layouts_fail_loudly(stages, match):
    with pytest.raises(ValueError, match=match):
        config(*stages)


def test_stage_cannot_exceed_model_or_training_maximum():
    with pytest.raises(ValueError, match="exceeds"):
        build(
            config(ContextStage(8, 10), ContextStage(64, None)),
            training_max=64,
            model_max=32,
        )


def test_first_stage_must_support_mtp_alignment():
    with pytest.raises(ValueError, match="MTP"):
        build(
            config(ContextStage(2, 10), ContextStage(8, None)),
            mtp_min=3,
        )


def test_each_sample_must_keep_enough_valid_tokens_for_mtp():
    curriculum = build(
        config(ContextStage(3, 10), ContextStage(8, None)),
        mtp_min=3,
    )
    curriculum.validate_valid_tokens(torch.tensor([4, 3]))
    with pytest.raises(ValueError, match="too few valid tokens for MTP"):
        curriculum.validate_valid_tokens(torch.tensor([4, 2]))


def test_transition_boundary_updates_length_and_never_moves_backward():
    curriculum = build(
        config(
            ContextStage(8, 10),
            ContextStage(16, 20),
            ContextStage(32, None),
        )
    )
    assert not curriculum.update(9)
    assert curriculum.current_max_seq_len() == 8
    assert curriculum.update(10)
    assert curriculum.stage_index == 1
    assert curriculum.current_max_seq_len() == 16
    assert curriculum.last_transition.old_max_seq_len == 8
    assert curriculum.last_transition.new_max_seq_len == 16
    with pytest.raises(ValueError, match="backward"):
        curriculum.update(9)


def test_multiple_thresholds_can_be_crossed_in_one_safe_transition():
    curriculum = build(
        config(
            ContextStage(4, 5),
            ContextStage(8, 10),
            ContextStage(16, 20),
            ContextStage(32, None),
        )
    )
    assert curriculum.update(25)
    assert curriculum.stage_index == 3
    assert curriculum.current_max_seq_len() == 32
    assert curriculum.state.transition_count == 1
    assert not curriculum.update(100)


def test_state_roundtrip_restores_exact_stage_length_tokens_and_counters():
    stages = (
        ContextStage(4, 5),
        ContextStage(8, 10),
        ContextStage(16, None),
    )
    source = build(config(*stages), training_max=16, model_max=16)
    source.update(7)
    state = source.state_dict()
    restored = build(config(*stages), training_max=16, model_max=16)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.metrics() == {
        "context/stage_index": 1.0,
        "context/max_seq_len": 8.0,
        "context/tokens_seen": 7.0,
        "context/transition_count": 1.0,
    }


def test_incompatible_stage_definition_fails_loudly():
    source = build(
        config(ContextStage(4, 5), ContextStage(8, None))
    )
    state = source.state_dict()
    incompatible = build(
        config(ContextStage(4, 6), ContextStage(8, None))
    )
    with pytest.raises(ValueError, match="stage definition"):
        incompatible.load_state_dict(state)


def test_preservation_flags_cannot_disable_training_state_continuity():
    with pytest.raises(ValueError, match="optimizer"):
        ContextCurriculumConfig(
            stages=(ContextStage(4, None),),
            preserve_optimizer_state=False,
        )
    with pytest.raises(ValueError, match="scheduler"):
        ContextCurriculumConfig(
            stages=(ContextStage(4, None),),
            preserve_scheduler_state=False,
        )
