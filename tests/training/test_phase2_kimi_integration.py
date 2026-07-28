import math

import torch
from torch.utils.data import DataLoader

from src import KimiK3, kimi_k3_cpu_tiny_config
from training import (
    DiagnosticsConfig,
    KimiDiagnosticCollector,
    TrainerState,
    train_one_epoch,
)
from training.optimizer import KimiOptimizerConfig, build_kimi_optimizer


def test_one_cpu_step_emits_finite_kimi_architecture_metrics_and_updates_qkv():
    torch.manual_seed(19)
    model = KimiK3(kimi_k3_cpu_tiny_config()).cpu()
    optimizer, registry = build_kimi_optimizer(
        model,
        KimiOptimizerConfig(
            adamw_lr=2e-4,
            muon_lr=2e-4,
            weight_decay=0.0,
            qk_clip_threshold=1e-12,
        ),
    )
    collector = KimiDiagnosticCollector(
        model,
        DiagnosticsConfig(
            standard_every_steps=1,
            deep_every_steps=100,
            sample_layers_per_standard_step=3,
        ),
        parameter_specs=registry.specs,
    )
    qkv_specs = [
        spec for spec in registry.specs
        if spec.optimizer_family == "per_head_muon"
    ]
    before = {
        spec.parameter_name: spec.parameter.detach().clone()
        for spec in qkv_specs
    }
    ids = torch.randint(5, 100, (2, 7))
    loader = DataLoader(
        [{
            "input_ids": row,
            "labels": row.clone(),
            "attention_mask": torch.ones(7, dtype=torch.bool),
        } for row in ids],
        batch_size=2,
    )
    captured = []
    state = TrainerState()
    stats = train_one_epoch(
        model, loader, optimizer, device="cpu", state=state,
        diagnostics=collector, grad_clip=1.0, use_mtp=True,
        on_optimizer_step=lambda metrics: captured.append(dict(metrics)),
    )
    assert math.isfinite(stats["loss"])
    assert state.optimizer_step == 1
    assert len(captured) == 1
    metrics = captured[0]
    assert any(name.startswith("kda/") for name in metrics)
    assert any(name.startswith("mla/") for name in metrics)
    assert any(name.startswith("moe/") for name in metrics)
    assert any(name.startswith("attnres/") for name in metrics)
    assert "mtp/hidden_rms" in metrics
    assert "mtp/gradient_rms" in metrics
    assert "mtp/update_rms" in metrics
    assert "qk_clip/events_step" in metrics
    assert metrics["qk_clip/events_step"] > 0
    numeric = {
        name: value for name, value in metrics.items()
        if name != "_alerts"
    }
    assert all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in numeric.values()
    )
    assert all(
        not torch.equal(before[spec.parameter_name], spec.parameter)
        for spec in qkv_specs
    )
