import json

import pytest
import torch

from training.diagnostics import (
    AlertManager,
    DiagnosticsConfig,
    KimiDiagnosticCollector,
    KimiTrainingPrinter,
)


def test_alert_patience_immediate_nonfinite_and_checkpoint_roundtrip():
    manager = AlertManager(patience_steps=2)
    assert manager.evaluate(
        {"moe/dead_expert_fraction_batch": 0.75}, 1
    ) == ()
    alerts = manager.evaluate(
        {"moe/dead_expert_fraction_batch": 0.75}, 2
    )
    assert [alert.code for alert in alerts] == [
        "MOE_DEAD_EXPERTS_PERSISTENT"
    ]
    critical = manager.evaluate({"train/loss_total": float("nan")}, 3)
    assert any(alert.severity == "critical" for alert in critical)
    state = manager.state_dict()
    restored = AlertManager(patience_steps=2)
    restored.load_state_dict(state)
    assert restored.state_dict() == state


@pytest.mark.parametrize(
    "metric,value,code",
    [
        ("block/layer_00/attention/branch_to_input_rms", 0.0,
         "INACTIVE_BLOCK"),
        ("attnres/source_entropy_normalized", 0.0,
         "ATTNRES_SINGLE_SOURCE_COLLAPSE"),
        ("attnres/source_entropy_normalized", 1.0,
         "ATTNRES_UNIFORM_COLLAPSE"),
        ("kda/layer_00/fraction_alpha_near_one", 1.0,
         "KDA_RETENTION_SATURATION"),
        ("kda/layer_00/fraction_beta_near_zero", 1.0,
         "KDA_STATE_WRITE_CLOSED"),
        ("kda/layer_00/state_growth_ratio", 3.0,
         "KDA_STATE_EXPLOSION"),
        ("mla/layer_00/output_gate_saturation_low", 1.0,
         "MLA_OUTPUT_GATE_CLOSED"),
        ("moe/layer_00/routed_to_total_ratio", 0.0,
         "MOE_ROUTED_BRANCH_INACTIVE"),
        ("optimizer/mtp/grad_rms", 0.0, "MTP_DISCONNECTED"),
        ("mtp/valid_tokens", 0.0, "MTP_NO_VALID_TOKENS"),
        ("qk_clip/fraction_layers_clipped", 1.0,
         "QK_CLIP_PERSISTENT"),
        ("representation/dead_feature_fraction", 1.0,
         "REPRESENTATION_VARIANCE_COLLAPSE"),
    ],
)
def test_each_collapse_detector_is_triggered_by_an_induced_pathology(
    metric, value, code
):
    manager = AlertManager(patience_steps=2)
    assert manager.evaluate({metric: value}, 1) == ()
    alerts = manager.evaluate({metric: value}, 2)
    assert code in {alert.code for alert in alerts}


def test_diagnostic_schedule_and_budget_degradation_keep_cheap_checks():
    config = DiagnosticsConfig(
        cheap_every_steps=1, standard_every_steps=2, deep_every_steps=4,
        max_diagnostic_time_fraction=0.01,
    )
    collector = KimiDiagnosticCollector(
        torch.nn.Linear(2, 2), config, parameter_specs=()
    )
    assert collector.level_for_step(1) == "cheap"
    assert collector.level_for_step(2) == "standard"
    collector.last_diagnostic_time_ms = 20
    collector.current_diagnostic_time_ms = 20
    collector.active_level = "standard"
    metrics, alerts = collector.capture_after_optimizer_step(
        step=2, step_time_ms=100
    )
    assert metrics["diagnostics/budget_exceeded"] == 1
    assert metrics["diagnostics/degradation_level"] == 1
    assert collector.level_for_step(4) == "cheap"
    assert any(
        alert.code == "DIAGNOSTIC_BUDGET_EXCEEDED" for alert in alerts
    )
    assert json.dumps(collector.diagnostic_snapshot(2))


def test_layer_rotation_is_deterministic_and_never_exceeds_limit():
    config = DiagnosticsConfig(
        standard_every_steps=2, sample_layers_per_standard_step=2
    )
    first = KimiDiagnosticCollector(torch.nn.Linear(2, 2), config)
    second = KimiDiagnosticCollector(torch.nn.Linear(2, 2), config)
    sample = first._sample_layer_indices(12, 8)
    assert sample == second._sample_layer_indices(12, 8)
    assert len(sample) == 2
    assert sample == tuple(sorted(sample))


def test_printer_uses_readable_blocks_not_single_unstructured_line(capsys):
    printer = KimiTrainingPrinter(width=70)
    printer.print_step(
        3,
        {
            "train/loss_total": 2.0, "train/loss_ntp": 1.5,
            "train/tokens_per_second": 200.0,
            "learning_rate": 3e-4,
            "grad_norm_pre_clip": 2.5,
            "moe/layer_00/routed_to_total_ratio": 0.7,
        },
    )
    output = capsys.readouterr().out
    assert "Optimizer step 000003" in output
    assert "Convergence" in output
    assert "Throughput & runtime" in output
    assert "Optimization health" in output
    assert "Architecture health" in output
    assert output.count("\n") > 10
