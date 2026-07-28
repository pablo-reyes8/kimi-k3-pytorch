import pytest
import torch

from src import GatedMLA, GatedMLAConfig
from training.optimizer import (
    KimiOptimizerConfig,
    QKClipController,
    build_kimi_optimizer,
)


def tiny_mla():
    return GatedMLA(GatedMLAConfig(
        d_model=12, num_heads=3, q_head_dim=4, v_head_dim=4,
        kv_latent_dim=5, attention_dropout=0.0, output_dropout=0.0,
    ))


def test_qk_clip_noop_below_threshold_and_exact_symmetric_rescale_above():
    model = tiny_mla()
    controller = QKClipController(model, threshold=4.0)
    query = model.projections.query.weight
    key = model.projections.latent_kv.key_up.weight
    value = model.projections.latent_kv.value_up.weight
    output = model.output_gate.output_proj.weight
    originals = [tensor.detach().clone() for tensor in (query, key, value, output)]

    model._last_qk_scale = torch.tensor(2.0)
    noop = controller.apply(1)
    assert not noop.applied
    for tensor, expected in zip((query, key, value, output), originals):
        torch.testing.assert_close(tensor, expected)

    model._last_qk_scale = torch.tensor(16.0)
    clipped = controller.apply(2)
    assert clipped.applied
    assert clipped.events == 1
    assert clipped.max_preclip_scale == 16.0
    assert clipped.max_postclip_scale == 4.0
    assert clipped.mean_rescale_factor == pytest.approx(0.25)
    torch.testing.assert_close(query, originals[0] * 0.5)
    torch.testing.assert_close(key, originals[1] * 0.5)
    torch.testing.assert_close(value, originals[2])
    torch.testing.assert_close(output, originals[3])
    assert not query.requires_grad or query.grad_fn is None


def test_qk_clip_state_roundtrip_and_counters():
    model = tiny_mla()
    model._last_qk_scale = torch.tensor(8.0)
    controller = QKClipController(model, threshold=2.0, every_steps=2)
    skipped = controller.apply(1)
    assert not skipped.applied
    assert controller.apply(2).events == 1
    state = controller.state_dict()
    restored = QKClipController(model, threshold=2.0, every_steps=2)
    restored.load_state_dict(state)
    assert restored.total_events == 1
    assert restored.consecutive_steps_active == 1


def test_qk_clip_rejects_nonfinite_control_proxy():
    model = tiny_mla()
    model._last_qk_scale = torch.tensor(float("inf"))
    with pytest.raises(FloatingPointError):
        QKClipController(model, threshold=2).apply(1)


def test_qk_clip_is_not_applied_when_hybrid_step_is_skipped():
    model = tiny_mla()
    optimizer, _ = build_kimi_optimizer(
        model,
        KimiOptimizerConfig(
            adamw_lr=1e-3, qk_clip_threshold=1.0
        ),
    )
    model._last_qk_scale = torch.tensor(10.0)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    next(model.parameters()).grad.reshape(-1)[0] = float("nan")
    query_before = model.projections.query.weight.detach().clone()
    report = optimizer.step()
    assert not report.executed
    assert not report.qk_clip_applied
    assert optimizer.qk_clip.total_events == 0
    torch.testing.assert_close(
        model.projections.query.weight, query_before
    )
