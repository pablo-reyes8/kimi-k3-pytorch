import math
from types import SimpleNamespace

import pytest
import torch

from src.mtp.outputs import MTPDiagnostics
from training.diagnostics import (
    compute_attnres_metrics,
    compute_block_contribution,
    compute_kda_metrics,
    compute_loss_metrics,
    compute_mla_metrics,
    compute_moe_metrics,
    compute_mtp_metrics,
)


def assert_metrics(metrics, expected):
    """Assert every emitted metric, including that no untested key slipped in."""

    assert metrics.keys() == expected.keys()
    for name, value in expected.items():
        assert isinstance(metrics[name], float), name
        assert metrics[name] == pytest.approx(value), name


def test_loss_metrics_measure_only_ntp_perplexity_and_mtp_ratio():
    metrics = compute_loss_metrics(
        total_loss=4.0,
        ntp_loss=2.0,
        mtp_loss=4.0,
        ntp_tokens=12,
        mtp_tokens=8,
        lambda_mtp=0.5,
    )
    assert_metrics(
        metrics,
        {
            "train/loss_total": 4.0,
            "train/loss_ntp": 2.0,
            "train/perplexity_ntp_clipped": math.exp(2.0),
            "train/valid_ntp_tokens": 12.0,
            "train/valid_mtp_tokens": 8.0,
            "mtp/loss_weight": 0.5,
            "train/loss_mtp": 4.0,
            "mtp/loss": 4.0,
            "mtp/perplexity_clipped": math.exp(4.0),
            "mtp/valid_tokens": 8.0,
            "mtp/main_to_mtp_loss_ratio": 0.5,
        },
    )
    clipped = compute_loss_metrics(
        total_loss=30, ntp_loss=30, mtp_loss=None,
        ntp_tokens=1, mtp_tokens=0,
    )
    assert clipped["train/perplexity_ntp_clipped"] == pytest.approx(
        math.exp(20)
    )
    assert "train/loss_mtp" not in clipped


def test_block_metrics_match_analytic_residual_example():
    x_in = torch.tensor([1.0, -1.0])
    branch = torch.tensor([1.0, 1.0])
    x_out = x_in + branch
    assert_metrics(
        compute_block_contribution(
            x_in, branch, x_out, prefix="block/test"
        ),
        {
            "block/test/branch_to_input_rms": 1.0,
            "block/test/state_change_ratio": 1.0,
            "block/test/branch_input_cosine": 0.0,
            "block/test/output_rms": math.sqrt(2.0),
            "block/test/output_absmax": 2.0,
            "block/test/output_zero_fraction": 0.5,
        },
    )


def test_attnres_metrics_measure_entropy_sources_and_depth_weights():
    stats = SimpleNamespace(
        normalized_entropy=torch.tensor(1.0),
        weight_entropy=torch.tensor(math.log(2.0)),
        max_weight=torch.tensor(0.75),
        embedding_weight=torch.tensor(0.25),
        current_partial_weight=torch.tensor(0.75),
        dominant_source_index=torch.tensor(1.0),
        mean_weights=torch.tensor([0.25, 0.75]),
    )
    depth = SimpleNamespace(site_stats=(stats,), final_output_stats=stats)
    assert_metrics(
        compute_attnres_metrics(depth),
        {
            "attnres/source_entropy_normalized": 1.0,
            "attnres/effective_num_sources": 2.0,
            "attnres/max_source_weight": 0.75,
            "attnres/embedding_source_weight": 0.25,
            "attnres/current_block_partial_weight": 0.75,
            "attnres/oldest_block_weight": 0.75,
            "attnres/top1_source_index_mean": 1.0,
            "attnres/source_weight_cv": 0.5,
        },
    )


def test_kda_metric_mapping_preserves_every_defined_quantity():
    source_names = {
        "alpha_mean": 1, "alpha_std": 2, "alpha_min": 3, "alpha_max": 4,
        "fraction_alpha_near_lower_bound": 5,
        "fraction_alpha_near_one": 6, "beta_mean": 7, "beta_std": 8,
        "beta_saturation_low": 9, "beta_saturation_high": 10,
        "log_decay_mean": 11, "cumulative_log_decay_min": 12,
        "cumulative_log_decay_mean": 13, "state_rms": 14,
        "state_absmax": 15, "recurrent_output_rms": 16,
        "gated_output_rms": 17, "output_gate_mean": 18,
        "output_gate_saturation_low": 19,
        "output_gate_saturation_high": 20,
    }
    expected_names = (
        "alpha_mean", "alpha_std", "alpha_min_sampled",
        "alpha_max_sampled", "fraction_alpha_near_lower_bound",
        "fraction_alpha_near_one", "beta_mean", "beta_std",
        "fraction_beta_near_zero", "fraction_beta_near_one",
        "log_decay_mean", "cumulative_log_decay_min",
        "cumulative_log_decay_mean", "state_rms", "state_absmax",
        "recurrent_output_rms", "gated_output_rms", "output_gate_mean",
        "output_gate_saturation_low", "output_gate_saturation_high",
    )
    expected = {
        f"kda/{name}": float(index)
        for name, index in zip(expected_names, range(1, 21))
    }
    assert_metrics(compute_kda_metrics(source_names), expected)


def test_mla_metric_mapping_includes_qkv_gate_attention_and_clip_state():
    source_names = {
        "attention_entropy_normalized": 1,
        "attention_max_probability": 2, "attention_output_rms": 3,
        "gated_output_rms": 4, "gate_mean": 5,
        "gate_saturation_low": 6, "gate_saturation_high": 7,
        "q_rms": 8, "k_rms": 9, "v_rms": 10, "qk_scale_max": 11,
    }
    public = (
        "attention_entropy_normalized_sampled",
        "max_attention_probability_sampled", "output_rms",
        "gated_output_rms", "output_gate_mean",
        "output_gate_saturation_low", "output_gate_saturation_high",
        "q_rms", "k_rms", "v_rms", "qk_scale_max",
    )
    expected = {
        f"mla/{name}": float(index)
        for name, index in zip(public, range(1, 12))
    }
    expected["mla/qk_clip_active"] = 1.0
    assert_metrics(
        compute_mla_metrics(source_names, qk_clip_active=True), expected
    )


def test_moe_metrics_have_correct_load_entropy_bias_and_branch_semantics():
    diagnostics = SimpleNamespace(
        mean_load=torch.tensor(1.0), min_load=torch.tensor(0.0),
        max_load=torch.tensor(2.0), std_load=torch.tensor(1.0),
        coefficient_of_variation=torch.tensor(1.0),
        expert_load=torch.tensor([0, 2]), num_assignments=4, num_tokens=2,
        zero_load_experts=torch.tensor(1),
        routing_entropy_over_selected=torch.tensor(math.log(2.0)),
        selected_weight_max=torch.tensor(0.75),
        routing_bias_mean=torch.tensor(0.1),
        routing_bias_std=torch.tensor(0.2),
        routing_bias_min=torch.tensor(-0.3),
        routing_bias_max=torch.tensor(0.4),
        qb_update_rms=torch.tensor(0.05),
        qb_quantile_error_estimate=torch.tensor(0.25),
        shared_output_rms=torch.tensor(2.0),
        routed_aggregate_rms_before_norm=torch.tensor(3.0),
        routed_aggregate_rms_after_norm=torch.tensor(1.0),
        routed_output_rms=torch.tensor(4.0),
        shared_to_total_ratio=torch.tensor(0.4),
        routed_to_total_ratio=torch.tensor(0.8),
        shared_routed_cosine=torch.tensor(-0.5),
        output_rms=torch.tensor(5.0),
    )
    assert_metrics(
        compute_moe_metrics(diagnostics),
        {
            "moe/load_mean": 1.0, "moe/load_std": 1.0,
            "moe/load_cv": 1.0, "moe/load_max_to_mean": 2.0,
            "moe/load_min_to_mean": 0.0,
            "moe/dead_expert_fraction_batch": 0.5,
            "moe/router_entropy_normalized": 1.0,
            "moe/top1_share": 0.75,
            "moe/topk_weight_entropy": math.log(2.0),
            "moe/qb_bias_mean": 0.1, "moe/qb_bias_std": 0.2,
            "moe/qb_bias_absmax": 0.4,
            "moe/qb_update_rms": 0.05,
            "moe/qb_quantile_error_estimate": 0.25,
            "moe/shared_output_rms": 2.0,
            "moe/routed_pre_norm_rms": 3.0,
            "moe/routed_post_norm_rms": 1.0,
            "moe/routed_up_projection_rms": 4.0,
            "moe/shared_to_total_ratio": 0.4,
            "moe/routed_to_total_ratio": 0.8,
            "moe/shared_routed_cosine": -0.5,
            "moe/output_rms": 5.0,
        },
    )


def test_mtp_metrics_do_not_confuse_hidden_rms_with_vector_norm():
    diagnostics = MTPDiagnostics(
        valid_token_count=torch.tensor(6),
        token_accuracy=torch.tensor(0.5),
        mean_logit_entropy=torch.tensor(1.5),
        mean_hidden_norm=torch.tensor(4.0),
        hidden_rms=torch.tensor(2.0),
        fusion_output_rms=torch.tensor(3.0),
        block={"layers": ({
            "input_norm": torch.tensor(2.0),
            "attention_output_norm": torch.tensor(1.0),
        },)},
    )
    assert_metrics(
        compute_mtp_metrics(diagnostics, mtp_loss=3.0, loss_weight=0.25),
        {
            "mtp/valid_tokens": 6.0, "mtp/token_accuracy": 0.5,
            "mtp/logit_entropy": 1.5, "mtp/hidden_rms": 2.0,
            "mtp/hidden_norm_mean": 4.0, "mtp/loss_weight": 0.25,
            "mtp/fusion_output_rms": 3.0,
            "mtp/block_branch_to_input_rms": 0.5,
            "mtp/loss": 3.0,
        },
    )
