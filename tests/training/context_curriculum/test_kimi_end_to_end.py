import math

import torch

from src import KimiK3, build_mtp_training_view, kimi_k3_cpu_tiny_config
from tests.kda.conftest import tiny_kda
from training import (
    ContextCurriculumConfig,
    ContextStage,
    ProgressiveContextCurriculum,
    TrainerState,
    build_adamw_optimizer,
    train_one_epoch,
)


def test_real_kimi_remains_finite_and_updates_after_context_transition():
    torch.manual_seed(31)
    model = KimiK3(
        kimi_k3_cpu_tiny_config(enable_vision=False)
    ).cpu()
    optimizer, _ = build_adamw_optimizer(
        model, learning_rate=2e-4, weight_decay=0
    )
    curriculum = ProgressiveContextCurriculum(
        ContextCurriculumConfig(
            enabled=True,
            reset_dataloader_on_transition=False,
            stages=(
                ContextStage(4, 8),
                ContextStage(8, None),
            ),
        ),
        training_max_seq_len=8,
        model_max_seq_len=8,
        mtp_min_seq_len=3,
    )
    first = torch.randint(5, 100, (2, 4))
    second = torch.randint(5, 100, (2, 8))
    callbacks = []
    tracked_after_transition = []

    def capture(metrics):
        callbacks.append(dict(metrics))
        if metrics["optimizer_step"] == 1:
            tracked_after_transition.append(
                model.mtp.fusion.projection.weight.detach().clone()
            )

    state = TrainerState()
    stats = train_one_epoch(
        model,
        [
            {"input_ids": first, "labels": first.clone()},
            {"input_ids": second, "labels": second.clone()},
        ],
        optimizer,
        state=state,
        curriculum=curriculum,
        use_mtp=True,
        stop_on_context_transition=False,
        on_optimizer_step=capture,
    )
    assert math.isfinite(stats["loss"])
    assert all(
        math.isfinite(metrics["train/loss_total"])
        for metrics in callbacks
    )
    assert [metrics["context/max_seq_len"] for metrics in callbacks] == [8, 8]
    assert callbacks[0]["context/old_max_seq_len"] == 4
    assert callbacks[0]["context/new_max_seq_len"] == 8
    assert not torch.equal(
        tracked_after_transition[0],
        model.mtp.fusion.projection.weight,
    )
    assert state.optimizer_step == 2
    assert state.tokens_seen == 24


def test_kda_recurrent_chunkwise_parity_at_each_representative_stage():
    model = tiny_kda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training_hidden = torch.randn(2, 8, 12)
    model(
        training_hidden, mode="chunkwise"
    ).hidden_states.square().mean().backward()
    optimizer.step()
    model.eval()
    for length in (4, 8):
        hidden = torch.randn(2, length, 12)
        with torch.no_grad():
            recurrent = model(hidden, mode="recurrent").hidden_states
            chunkwise = model(hidden, mode="chunkwise").hidden_states
        torch.testing.assert_close(
            recurrent, chunkwise, atol=3e-5, rtol=3e-5
        )


def test_mtp_alignment_remains_exact_at_512_and_1k_contexts():
    for length in (512, 1024):
        hidden = torch.randn(1, length, 4)
        ids = torch.arange(length).remainder(127)[None]
        mask = torch.ones(1, length, dtype=torch.bool)
        view = build_mtp_training_view(hidden, ids, mask, ids)
        assert view.source_hidden.shape == (1, length - 2, 4)
        assert view.future_input_ids.shape == (1, length - 2)
        assert view.target_ids.shape == (1, length - 2)
        assert view.valid_mask.sum() == length - 2
        assert torch.equal(view.future_input_ids, ids[:, 1:-1])
        assert torch.equal(view.target_ids, ids[:, 2:])


def test_tiny_multimodal_batch_overfits_and_keeps_vision_gradients():
    torch.manual_seed(123)
    model = KimiK3(
        kimi_k3_cpu_tiny_config(enable_mtp=False)
    ).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=0
    )
    ids = torch.tensor([[1, 3, 3, 3, 3, 20, 21, 2]])
    labels = ids.clone()
    labels[:, 1:5] = -100
    mask = torch.ones_like(ids, dtype=torch.bool)
    pixels = torch.randn(1, 3, 16, 16)
    losses = []
    saw_vision_gradient = False
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            ids,
            mask,
            pixel_values=pixels,
            image_counts=torch.ones(1, dtype=torch.long),
            labels=labels,
        )
        losses.append(float(output.loss.detach()))
        output.loss.backward()
        saw_vision_gradient |= any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad))
            for parameter in model.vision_encoder.parameters()
        )
        optimizer.step()
    assert saw_vision_gradient
    assert losses[-1] < losses[0] - 0.2
