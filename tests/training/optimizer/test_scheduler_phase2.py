import pytest
import torch

from training import build_warmup_cosine_scheduler
from training.optimizer import KimiOptimizerConfig, build_kimi_optimizer


def test_first_optimizer_update_uses_first_warmup_lr():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps=10, warmup_steps=2,
        min_lr=0.0, prepare_first_update=True,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_hybrid_scheduler_updates_and_restores_both_learning_rates():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4))
    optimizer, _ = build_kimi_optimizer(
        model,
        KimiOptimizerConfig(
            kind="muon_adamw", adamw_lr=0.1, muon_lr=0.2,
            qk_clip_enabled=False,
        ),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps=10, warmup_steps=2,
        min_lr=0.01, min_muon_lr=0.02, prepare_first_update=True,
    )
    assert optimizer.adamw.param_groups[0]["lr"] == pytest.approx(0.05)
    assert optimizer.muon.param_groups[0]["lr"] == pytest.approx(0.1)
    scheduler.step()
    state = scheduler.state_dict()
    assert optimizer.adamw.param_groups[0]["lr"] == pytest.approx(0.1)
    assert optimizer.muon.param_groups[0]["lr"] == pytest.approx(0.2)

    restored = build_warmup_cosine_scheduler(
        optimizer, total_steps=10, warmup_steps=2,
        min_lr=0.01, min_muon_lr=0.02, prepare_first_update=True,
    )
    restored.load_state_dict(state)
    assert restored.get_last_lr() == pytest.approx(scheduler.get_last_lr())
