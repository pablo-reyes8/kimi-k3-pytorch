from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training import (
    CheckpointConfig,
    ContextCurriculumConfig,
    ContextStage,
    OptimizerConfig,
    ProgressiveContextCurriculum,
    TrainerState,
    TrainingConfig,
    WarmupCosineLR,
    train_kimiK3,
    train_one_epoch,
)
from training.context_curriculum import build_context_loader


@dataclass(frozen=True)
class ToyConfig:
    max_seq_len: int


class ToyTokenLM(torch.nn.Module):
    def __init__(self, max_seq_len=1024, vocab_size=16, width=4):
        super().__init__()
        self.config = ToyConfig(max_seq_len)
        self.embedding = torch.nn.Embedding(vocab_size, width)
        self.projection = torch.nn.Linear(width, vocab_size)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        del kwargs
        logits = self.projection(self.embedding(input_ids))
        if labels is None:
            return SimpleNamespace(logits=logits, loss=None)
        valid_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
        valid_labels = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(valid_logits, valid_labels)
        return SimpleNamespace(logits=logits, loss=loss, loss_output=None)


def pcc(*stages, reset=True):
    return ContextCurriculumConfig(
        enabled=True,
        stages=tuple(stages),
        reset_dataloader_on_transition=reset,
    )


def batch(length, *, batch_size=1):
    ids = (
        torch.arange(length).remainder(15).add(1)
        .repeat(batch_size, 1)
    )
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": torch.ones_like(ids, dtype=torch.bool),
    }


def test_transition_occurs_only_after_complete_gradient_accumulation_window():
    model = ToyTokenLM(max_seq_len=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineLR(
        optimizer, total_steps=2, warmup_steps=0
    )
    curriculum = ProgressiveContextCurriculum(
        pcc(ContextStage(4, 4), ContextStage(8, None), reset=False),
        training_max_seq_len=8,
        model_max_seq_len=8,
    )
    callbacks = []
    state = TrainerState()
    stats = train_one_epoch(
        model,
        [batch(4), batch(4)],
        optimizer,
        scheduler=scheduler,
        grad_accum_steps=2,
        state=state,
        curriculum=curriculum,
        stop_on_context_transition=False,
        on_optimizer_step=lambda metrics: callbacks.append(dict(metrics)),
    )
    assert stats["optimizer_steps"] == 1
    assert state.micro_step == 2
    assert state.optimizer_step == 1
    assert state.tokens_seen == 8
    assert scheduler.step_num == 1
    assert len(callbacks) == 1
    assert callbacks[0]["context/old_max_seq_len"] == 4
    assert callbacks[0]["context/new_max_seq_len"] == 8


def test_transition_preserves_optimizer_and_scheduler_state():
    model = ToyTokenLM(max_seq_len=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineLR(
        optimizer, total_steps=2, warmup_steps=0
    )
    curriculum = ProgressiveContextCurriculum(
        pcc(ContextStage(4, 4), ContextStage(8, None), reset=False),
        training_max_seq_len=8,
        model_max_seq_len=8,
    )
    optimizer_identity = id(optimizer)
    scheduler_identity = id(scheduler)
    train_one_epoch(
        model,
        [batch(4), batch(8)],
        optimizer,
        scheduler=scheduler,
        curriculum=curriculum,
        stop_on_context_transition=False,
    )
    assert id(optimizer) == optimizer_identity
    assert id(scheduler) == scheduler_identity
    assert scheduler.step_num == 2
    assert all(
        state["step"] == 2
        for state in optimizer.state.values()
        if "step" in state
    )


def test_master_orchestrator_runs_real_512_to_1k_loader_rebuild(tmp_path):
    calls = []

    def loader_factory(max_seq_len):
        calls.append(max_seq_len)
        return DataLoader([batch(max_seq_len)["input_ids"][0]], batch_size=1)

    result = train_kimiK3(
        model=ToyTokenLM(),
        train_loader_factory=loader_factory,
        device="cpu",
        training_config=TrainingConfig(
            epochs=1,
            precision="fp32",
            use_mtp=False,
            prediction_every_epochs=None,
            max_seq_len=1024,
            log_every_steps=1,
        ),
        optimizer_config=OptimizerConfig(
            learning_rate=1e-3, weight_decay=0
        ),
        context_curriculum_config=pcc(
            ContextStage(512, 512),
            ContextStage(1024, None),
        ),
        checkpoint_config=CheckpointConfig(
            output_dir=tmp_path, run_name="pcc"
        ),
        verbose=False,
    )
    assert calls == [512, 1024]
    assert result["curriculum"].current_max_seq_len() == 1024
    assert result["curriculum"].state.transition_count == 1
    assert result["state"].optimizer_step == 2
    assert result["state"].tokens_seen == 1536
    assert result["scheduler"].step_num == 2
    assert torch.isfinite(
        torch.tensor(result["history"]["train"][0]["loss"])
    )


def test_enabled_loader_reset_requires_explicit_factory():
    with pytest.raises(ValueError, match="train_loader_factory"):
        train_kimiK3(
            model=ToyTokenLM(max_seq_len=8),
            train_loader=[batch(4)],
            device="cpu",
            training_config=TrainingConfig(
                precision="fp32", use_mtp=False, max_seq_len=8,
                prediction_every_epochs=None,
            ),
            optimizer_config=OptimizerConfig(),
            context_curriculum_config=pcc(
                ContextStage(4, 4), ContextStage(8, None)
            ),
            verbose=False,
        )


def test_master_rejects_ambiguous_loader_and_factory_inputs():
    model = ToyTokenLM()
    loader = [{"input_ids": torch.ones(1, 4, dtype=torch.long)}]
    with pytest.raises(ValueError, match="not both"):
        train_kimiK3(
            model=model,
            train_loader=loader,
            train_loader_factory=lambda max_seq_len: loader,
            training_config=TrainingConfig(
                epochs=1, precision="fp32", use_mtp=False
            ),
            verbose=False,
        )


def test_loader_factory_can_receive_context_as_keyword_contract():
    received = []

    def factory(**kwargs):
        received.append(kwargs)
        return ["loader"]

    assert build_context_loader(factory, 512) == ["loader"]
    assert received == [{"max_seq_len": 512}]


def test_context_metrics_measure_padding_tokens_speed_and_transition():
    model = ToyTokenLM(max_seq_len=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    curriculum = ProgressiveContextCurriculum(
        pcc(ContextStage(4, 7), ContextStage(8, None), reset=False),
        training_max_seq_len=8,
        model_max_seq_len=8,
    )
    padded = batch(4, batch_size=2)
    padded["attention_mask"][0, -1] = False
    padded["labels"][0, -1] = -100
    captured = []
    train_one_epoch(
        model, [padded], optimizer, curriculum=curriculum,
        stop_on_context_transition=False,
        on_optimizer_step=lambda metrics: captured.append(dict(metrics)),
    )
    metrics = captured[0]
    assert metrics["context/valid_tokens_per_step"] == 7
    assert metrics["context/padding_fraction"] == pytest.approx(1 / 8)
    assert metrics["context/stage_index"] == 1
    assert metrics["context/max_seq_len"] == 8
    assert metrics["context/tokens_seen_at_transition"] == 7
    assert metrics["context/tokens_per_second"] > 0
    assert metrics["context/step_time_seconds"] > 0
    assert metrics["context/peak_memory_mb"] == 0
