import math
from types import SimpleNamespace

import pytest
import torch

from src.kda.diagnostics import build_kda_diagnostics
from src.mla.diagnostics import build_mla_diagnostics
from src.mtp.head import KimiMTPHead
from src.stable_latent_moe.diagnostics import build_moe_diagnostics
from src.stable_latent_moe.outputs import RouterOutput


def test_kda_source_diagnostics_measure_controlled_gates_decay_and_state():
    q = torch.ones(1, 2, 1, 1)
    k = -torch.ones_like(q)
    beta = torch.tensor([[[0.0], [1.0]]])
    alpha = torch.tensor([[[[0.2]], [[1.0]]]])
    g = alpha.log()
    state = torch.tensor([[[[3.0]]]])
    recurrent = torch.full((1, 2, 1, 1), 2.0)
    gated = torch.tensor([[[0.0], [4.0]]])
    gate = torch.tensor([[[0.0], [1.0]]])
    metrics = build_kda_diagnostics(
        q, k, beta, g, alpha, state, recurrent, gated, gate,
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        chunk_size=1, alpha_lower_bound=0.2,
    )
    assert metrics["alpha_mean"] == pytest.approx(0.6)
    assert metrics["alpha_std"] == pytest.approx(0.4)
    assert metrics["fraction_alpha_near_lower_bound"] == pytest.approx(0.5)
    assert metrics["fraction_alpha_near_one"] == pytest.approx(0.5)
    assert metrics["beta_mean"] == pytest.approx(0.5)
    assert metrics["beta_saturation_low"] == pytest.approx(0.5)
    assert metrics["beta_saturation_high"] == pytest.approx(0.5)
    assert metrics["state_rms"] == pytest.approx(3.0)
    assert metrics["state_absmax"] == pytest.approx(3.0)
    assert metrics["recurrent_output_rms"] == pytest.approx(2.0)
    assert metrics["gated_output_rms"] == pytest.approx(math.sqrt(8))
    assert metrics["output_gate_saturation"] == pytest.approx(1.0)
    assert all(value.ndim == 0 and not value.requires_grad
               for value in metrics.values())


def test_mla_source_diagnostics_measure_uniform_entropy_and_qkv_rms():
    query = torch.ones(1, 2, 1, 2)
    key = torch.full_like(query, 2.0)
    value = torch.full_like(query, 3.0)
    latent = torch.ones(1, 2, 3)
    gate = torch.tensor([[[0.0, 1.0], [0.5, 0.5]]])
    attention = torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    diagnostics = build_mla_diagnostics(
        query, key, value, latent, gate, attention, mask, mask,
        cache_elements=6, full_kv_width=6,
        attention_output=torch.full((1, 2, 1, 2), 4.0),
        final_output=torch.full((1, 2, 2), 5.0),
        qk_scale=torch.tensor(7.0),
    )
    assert diagnostics["attention_entropy_normalized"] == pytest.approx(0.5)
    assert diagnostics["attention_max_probability"] == pytest.approx(0.75)
    assert diagnostics["q_rms"] == pytest.approx(1.0)
    assert diagnostics["k_rms"] == pytest.approx(2.0)
    assert diagnostics["v_rms"] == pytest.approx(3.0)
    assert diagnostics["attention_output_rms"] == pytest.approx(4.0)
    assert diagnostics["gated_output_rms"] == pytest.approx(5.0)
    assert diagnostics["qk_scale_max"] == pytest.approx(7.0)


def test_mtp_source_diagnostics_compute_true_elementwise_hidden_rms():
    logits = torch.tensor([[[4.0, 0.0], [0.0, 4.0]]])
    hidden = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    view = SimpleNamespace(
        valid_mask=torch.tensor([[True, False]]),
        target_ids=torch.tensor([[0, 1]]),
    )
    fusion = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    diagnostics = KimiMTPHead._diagnostics(
        logits, hidden, fusion, view, None
    )
    assert diagnostics.valid_token_count == 1
    assert diagnostics.token_accuracy == 1
    assert diagnostics.mean_hidden_norm == pytest.approx(5.0)
    assert diagnostics.hidden_rms == pytest.approx(math.sqrt(12.5))
    assert diagnostics.fusion_output_rms == pytest.approx(math.sqrt(2.5))


def test_moe_source_qb_metrics_measure_bias_delta_and_load_quantile_error():
    router = RouterOutput(
        selected_experts=torch.tensor([[0, 0], [0, 1]]),
        selected_raw_scores=torch.ones(2, 2),
        selected_weights=torch.full((2, 2), 0.5),
        expert_load=torch.tensor([3, 1]),
        routing_bias_before=torch.tensor([0.0, 0.0]),
        routing_bias_after=torch.tensor([0.3, 0.4]),
    )
    shared = torch.ones(2, 2)
    routed = torch.full((2, 2), 2.0)
    diagnostics = build_moe_diagnostics(
        router, num_shared_experts=1, shared_output=shared,
        routed_output=routed, aggregate=routed,
        normalized_aggregate=torch.ones_like(routed),
    )
    assert diagnostics.qb_update_rms == pytest.approx(
        math.sqrt((0.3 ** 2 + 0.4 ** 2) / 2)
    )
    # Loads [3/4, 1/4] versus uniform target [1/2, 1/2].
    assert diagnostics.qb_quantile_error_estimate == pytest.approx(0.25)
    assert diagnostics.shared_to_total_ratio == pytest.approx(1 / 3)
    assert diagnostics.routed_to_total_ratio == pytest.approx(2 / 3)
    assert diagnostics.shared_routed_cosine == pytest.approx(1.0)
