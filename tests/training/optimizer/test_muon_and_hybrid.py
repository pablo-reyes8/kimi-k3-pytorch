from types import SimpleNamespace
import copy

import pytest
import torch

from training.optimizer import (
    KimiMuon,
    KimiOptimizerConfig,
    build_kimi_optimizer,
    zeropower_via_newton_schulz,
)


def make_muon(parameter, **overrides):
    values = dict(
        lr=0.1, momentum=0.0, nesterov=False, ns_steps=2, eps=1e-7,
        weight_decay=0.0, update_rms_scaling=False,
        spec_by_parameter={
            id(parameter): SimpleNamespace(optimizer_family="muon")
        },
    )
    values.update(overrides)
    return KimiMuon([parameter], **values)


def test_muon_one_step_matches_hand_reference_and_emits_plain_metrics():
    parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    gradient = torch.tensor([[0.5, -1.0], [2.0, 0.25]])
    parameter.grad = gradient.clone()
    optimizer = make_muon(parameter)
    expected_update = zeropower_via_newton_schulz(gradient, steps=2)
    expected = parameter.detach() - 0.1 * expected_update
    metrics = optimizer.step()
    torch.testing.assert_close(parameter, expected)
    assert metrics.keys() == {
        "muon/raw_update_rms", "muon/orthogonal_update_rms",
        "muon/scaled_update_rms", "muon/ns_gain",
        "muon/update_to_parameter_ratio",
    }
    assert all(isinstance(value, float) for value in metrics.values())


def test_muon_momentum_nesterov_decay_and_state_roundtrip():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    parameter.grad = torch.ones_like(parameter)
    optimizer = make_muon(
        parameter, momentum=0.5, nesterov=True, weight_decay=0.2
    )
    optimizer.step()
    assert torch.equal(
        optimizer.state[parameter]["momentum_buffer"], torch.ones(2, 2)
    )
    state = optimizer.state_dict()
    clone = torch.nn.Parameter(parameter.detach().clone())
    restored = make_muon(
        clone, momentum=0.5, nesterov=True, weight_decay=0.2
    )
    restored.load_state_dict(state)
    torch.testing.assert_close(
        restored.state[clone]["momentum_buffer"],
        optimizer.state[parameter]["momentum_buffer"],
    )


def test_muon_nonfinite_fails_before_parameter_or_state_mutation():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    parameter.grad = torch.tensor([[float("nan"), 0], [0, 0]])
    optimizer = make_muon(parameter)
    before = parameter.detach().clone()
    with pytest.raises(FloatingPointError):
        optimizer.step()
    torch.testing.assert_close(parameter, before)
    assert optimizer.state == {}


def test_hybrid_step_is_transactional_and_updates_muon_and_adamw():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    config = KimiOptimizerConfig(
        kind="muon_adamw", adamw_lr=1e-2, muon_lr=2e-2,
        weight_decay=0.0, qk_clip_enabled=False,
    )
    optimizer, registry = build_kimi_optimizer(model, config)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    model(torch.ones(2, 3)).sum().backward()
    report = optimizer.step()
    assert report.executed
    assert all(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )
    assert registry.missing == ()
    optimizer.zero_grad()
    assert all(parameter.grad is None for parameter in model.parameters())

    model(torch.ones(2, 3)).sum().backward()
    first_matrix = next(
        spec.parameter for spec in registry.specs
        if spec.optimizer_family == "muon"
    )
    first_matrix.grad[0, 0] = float("nan")
    frozen = {name: value.detach().clone() for name, value in model.named_parameters()}
    skipped = optimizer.step()
    assert not skipped.executed
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, frozen[name])


def test_hybrid_state_roundtrip_reproduces_the_exact_next_update():
    torch.manual_seed(4)
    source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    config = KimiOptimizerConfig(
        kind="muon_adamw", adamw_lr=1e-2, muon_lr=2e-2,
        qk_clip_enabled=False,
    )
    source_optimizer, _ = build_kimi_optimizer(source, config)
    batch = torch.randn(2, 3)
    source(batch).square().mean().backward()
    source_optimizer.step()
    source_optimizer.zero_grad()

    resumed = copy.deepcopy(source)
    resumed_optimizer, _ = build_kimi_optimizer(resumed, config)
    resumed_optimizer.load_state_dict(
        copy.deepcopy(source_optimizer.state_dict())
    )

    source(batch).square().mean().backward()
    resumed(batch).square().mean().backward()
    source_report = source_optimizer.step()
    resumed_report = resumed_optimizer.step()
    for expected, actual in zip(source.parameters(), resumed.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert source_report.update_metrics == pytest.approx(
        resumed_report.update_metrics
    )
    assert source_optimizer.step_number == resumed_optimizer.step_number
