import torch
from torch.utils.data import DataLoader

from training import (
    CheckpointConfig,
    ContextCurriculumConfig,
    ContextStage,
    OptimizerConfig,
    ProgressiveContextCurriculum,
    TrainerState,
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    train_kimiK3,
)
from tests.training.context_curriculum.test_trainer_integration import (
    ToyTokenLM,
    batch,
)


def curriculum_config():
    return ContextCurriculumConfig(
        enabled=True,
        stages=(
            ContextStage(4, 4),
            ContextStage(8, None),
        ),
    )


def make_curriculum():
    return ProgressiveContextCurriculum(
        curriculum_config(),
        training_max_seq_len=8,
        model_max_seq_len=8,
    )


def test_checkpoint_contains_canonical_context_curriculum_key(tmp_path):
    model = ToyTokenLM(max_seq_len=8)
    curriculum = make_curriculum()
    curriculum.update(4)
    path = save_checkpoint(
        tmp_path / "curriculum.pt",
        model,
        trainer_state=TrainerState(tokens_seen=4, curriculum_stage_index=1),
        curriculum=curriculum,
    )
    payload = torch.load(path, weights_only=False)
    assert payload["context_curriculum"] == curriculum.state_dict()
    assert payload["context_curriculum"]["max_seq_len"] == 8
    assert payload["context_curriculum"]["tokens_seen"] == 4


def test_checkpoint_roundtrip_restores_exact_curriculum_and_trainer_state(tmp_path):
    source_model = ToyTokenLM(max_seq_len=8)
    source_curriculum = make_curriculum()
    source_curriculum.update(4)
    source_state = TrainerState(
        optimizer_step=1,
        tokens_seen=4,
        curriculum_stage_index=1,
    )
    path = save_checkpoint(
        tmp_path / "roundtrip.pt",
        source_model,
        trainer_state=source_state,
        curriculum=source_curriculum,
    )
    restored_model = ToyTokenLM(max_seq_len=8)
    restored_curriculum = make_curriculum()
    restored_state = TrainerState()
    loaded = load_checkpoint(
        path,
        restored_model,
        trainer_state=restored_state,
        curriculum=restored_curriculum,
    )
    assert restored_curriculum.state_dict() == source_curriculum.state_dict()
    assert restored_state.state_dict() == source_state.state_dict()
    assert loaded["curriculum_state"] == source_curriculum.state_dict()


def test_master_resume_rebuilds_loader_at_restored_length(tmp_path):
    first_calls = []

    def first_factory(max_seq_len):
        first_calls.append(max_seq_len)
        return DataLoader([batch(max_seq_len)["input_ids"][0]], batch_size=1)

    first = train_kimiK3(
        model=ToyTokenLM(max_seq_len=8),
        train_loader_factory=first_factory,
        device="cpu",
        training_config=TrainingConfig(
            epochs=1, precision="fp32", use_mtp=False,
            prediction_every_epochs=None, max_seq_len=8,
        ),
        optimizer_config=OptimizerConfig(learning_rate=1e-3),
        context_curriculum_config=curriculum_config(),
        checkpoint_config=CheckpointConfig(
            output_dir=tmp_path, run_name="first"
        ),
        verbose=False,
    )
    assert first_calls == [4, 8]

    resumed_calls = []

    def resumed_factory(max_seq_len):
        resumed_calls.append(max_seq_len)
        return DataLoader([batch(max_seq_len)["input_ids"][0]], batch_size=1)

    resumed = train_kimiK3(
        model=ToyTokenLM(max_seq_len=8),
        train_loader_factory=resumed_factory,
        device="cpu",
        training_config=TrainingConfig(
            epochs=2, precision="fp32", use_mtp=False,
            prediction_every_epochs=None, max_seq_len=8,
        ),
        optimizer_config=OptimizerConfig(learning_rate=1e-3),
        context_curriculum_config=curriculum_config(),
        checkpoint_config=CheckpointConfig(
            output_dir=tmp_path,
            run_name="resumed",
            resume_from=first["last_checkpoint"],
        ),
        verbose=False,
    )
    # Construction probes stage 0; resume then rebuilds at exact stage 1.
    assert resumed_calls[:2] == [4, 8]
    assert resumed["curriculum"].stage_index == 1
    assert resumed["curriculum"].current_max_seq_len() == 8
    assert resumed["state"].tokens_seen == 20


def test_checkpoint_with_incompatible_stages_fails_loudly(tmp_path):
    model = ToyTokenLM(max_seq_len=8)
    curriculum = make_curriculum()
    path = save_checkpoint(
        tmp_path / "bad_stages.pt", model, curriculum=curriculum
    )
    incompatible = ProgressiveContextCurriculum(
        ContextCurriculumConfig(
            enabled=True,
            stages=(ContextStage(4, 5), ContextStage(8, None)),
        ),
        training_max_seq_len=8,
        model_max_seq_len=8,
    )
    try:
        load_checkpoint(
            path,
            ToyTokenLM(max_seq_len=8),
            curriculum=incompatible,
        )
    except ValueError as error:
        assert "stage definition" in str(error)
    else:
        raise AssertionError("incompatible PCC checkpoint was accepted")
