import math

import pytest
import torch

from src import BaselineCausalLM, BaselineCausalLMConfig
from training.muon_optimizer import build_muon_adamw_optimizer
from training.scheduler import WarmupCosineLR, build_warmup_cosine_scheduler


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_steps": 0, "warmup_steps": 0},
        {"total_steps": 10, "warmup_steps": -1},
        {"total_steps": 10, "warmup_steps": 1, "min_lr": -0.1},
        {"total_steps": 10, "warmup_steps": 1, "min_muon_lr": -0.1},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    with pytest.raises(ValueError):
        WarmupCosineLR(optimizer, **kwargs)


def test_exact_linear_warmup_cosine_and_floor_values():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupCosineLR(
        optimizer, total_steps=6, warmup_steps=2, min_lr=0.2
    )
    observed = []
    for _ in range(7):
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    expected = [
        0.5,
        1.0,
        0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * 0.25)),
        0.6,
        0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * 0.75)),
        0.2,
        0.2,
    ]
    assert observed == pytest.approx(expected)


def test_zero_warmup_starts_on_cosine_curve():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    scheduler = WarmupCosineLR(optimizer, total_steps=4, warmup_steps=0)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        0.5 * (1 + math.cos(math.pi / 4))
    )


def test_set_step_and_negative_step_contract():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    scheduler = WarmupCosineLR(optimizer, total_steps=10, warmup_steps=2)
    scheduler.set_step(7)
    assert scheduler.step_num == 7
    with pytest.raises(ValueError):
        scheduler.set_step(-1)


def test_multiple_standard_parameter_groups_preserve_individual_base_rates():
    first, second = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD(
        [{"params": [first], "lr": 1.0}, {"params": [second], "lr": 0.5}]
    )
    scheduler = WarmupCosineLR(optimizer, total_steps=4, warmup_steps=2)
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.5, 0.25])
    assert scheduler.get_lr_dict()["lrs"] == pytest.approx([0.5, 0.25])


def test_state_roundtrip_restores_schedule_and_current_lr():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    scheduler = WarmupCosineLR(
        optimizer, total_steps=10, warmup_steps=2, min_lr=0.1
    )
    for _ in range(4):
        scheduler.step()
    state = scheduler.state_dict()
    restored = WarmupCosineLR(optimizer, total_steps=99, warmup_steps=0)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.get_last_lr() == scheduler.get_last_lr()


def tiny_model():
    return BaselineCausalLM(
        BaselineCausalLMConfig(
            vocab_size=32,
            d_model=16,
            n_layers=1,
            n_heads=2,
            mlp_hidden_dim=32,
            max_seq_len=8,
        )
    )


def test_hybrid_muon_and_adamw_receive_independent_lr_curves():
    optimizer, _ = build_muon_adamw_optimizer(
        tiny_model(), learning_rate=1e-3, muon_lr=1e-2
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=4,
        warmup_steps=2,
        min_lr=1e-4,
        min_muon_lr=1e-3,
    )
    scheduler.step()
    info = scheduler.get_lr_dict()
    assert info["muon_lr"] == pytest.approx(5e-3)
    assert info["adamw_lr"] == pytest.approx(5e-4)
    scheduler.set_step(4)
    info = scheduler.get_lr_dict()
    assert info["muon_lr"] == pytest.approx(1e-3)
    assert info["adamw_lr"] == pytest.approx(1e-4)
