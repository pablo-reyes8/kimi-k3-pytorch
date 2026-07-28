import copy

import pytest
import torch

from src.stable_latent_moe import (
    ExactQuantileBalancer,
    HistogramQuantileBalancer,
)
from tests.stable_latent_moe.conftest import tiny_moe


def scores_and_cutoffs(tokens=32):
    torch.manual_seed(229)
    scores = torch.rand(tokens, 4, dtype=torch.float64) * 0.8 + 0.1
    old_bias = torch.tensor([0.2, -0.1, 0.05, -0.15], dtype=torch.float64)
    cutoff = torch.topk(scores + old_bias, 3, dim=-1).values[:, 2]
    return scores, cutoff, old_bias


def test_exact_qb_matches_manual_quantile_without_adding_old_bias_twice():
    scores, cutoff, old_bias = scores_and_cutoffs()
    balancer = ExactQuantileBalancer(4, 2)
    update = balancer.compute_next_bias(scores, cutoff, old_bias)
    margins = scores - cutoff[:, None]
    quantiles = torch.quantile(margins, 0.5, dim=0)
    provisional = -quantiles
    expected = provisional - provisional.mean()
    torch.testing.assert_close(update.quantiles, quantiles)
    torch.testing.assert_close(update.next_bias, expected)
    torch.testing.assert_close(
        update.next_bias.mean(),
        torch.tensor(0.0, dtype=torch.float64),
        atol=2e-17,
        rtol=0,
    )


def test_exact_compute_is_pure_token_permutation_invariant_and_expert_equivariant():
    scores, cutoff, old_bias = scores_and_cutoffs()
    balancer = ExactQuantileBalancer(4, 2)
    snapshot = old_bias.clone()
    baseline = balancer.compute_next_bias(scores, cutoff, old_bias)
    permutation = torch.randperm(scores.shape[0])
    tokens_permuted = balancer.compute_next_bias(
        scores[permutation], cutoff[permutation], old_bias
    )
    torch.testing.assert_close(
        tokens_permuted.next_bias, baseline.next_bias
    )
    expert_permutation = torch.tensor([2, 0, 3, 1])
    experts_permuted = balancer.compute_next_bias(
        scores[:, expert_permutation],
        cutoff,
        old_bias[expert_permutation],
    )
    torch.testing.assert_close(
        experts_permuted.next_bias,
        baseline.next_bias[expert_permutation],
    )
    torch.testing.assert_close(old_bias, snapshot)


def test_centering_common_offset_cannot_change_routes():
    scores, _, old_bias = scores_and_cutoffs()
    provisional = old_bias + 13.0
    centered = provisional - provisional.mean()
    torch.testing.assert_close(
        torch.topk(scores + provisional, 2, dim=-1).indices,
        torch.topk(scores + centered, 2, dim=-1).indices,
    )


def test_qb_balances_the_controlled_eight_token_four_expert_example():
    generator = torch.Generator().manual_seed(0)
    scores = torch.rand(8, 4, generator=generator) * 0.8 + 0.1
    scores[:, 0] += 0.25
    old_bias = torch.zeros(4)
    top_two = torch.topk(scores + old_bias, 2, dim=-1)
    before = torch.bincount(
        top_two.indices[:, 0], minlength=4
    )
    update = ExactQuantileBalancer(4, 1).compute_next_bias(
        scores, top_two.values[:, 1], old_bias
    )
    after = torch.bincount(
        torch.topk(scores + update.next_bias, 1, dim=-1).indices.reshape(-1),
        minlength=4,
    )
    assert before.tolist() == [4, 0, 1, 3]
    assert after.tolist() == [2, 2, 2, 2]


@pytest.mark.parametrize("bins", [16, 64, 256])
def test_histogram_counts_every_margin_and_approximates_exact_within_bin(bins):
    scores, cutoff, old_bias = scores_and_cutoffs(tokens=128)
    histogram = HistogramQuantileBalancer(4, 2, bins, -1.0, 1.0).double()
    histogram.accumulate(scores[:37], cutoff[:37])
    histogram.accumulate(scores[37:], cutoff[37:])
    assert histogram.counts.dtype == torch.int64
    assert torch.equal(
        histogram.counts.sum(-1),
        torch.full((4,), 128, dtype=torch.int64),
    )
    approximate = histogram.compute_next_bias(old_bias)
    exact = ExactQuantileBalancer(4, 2).compute_next_bias(
        scores, cutoff, old_bias
    )
    torch.testing.assert_close(
        approximate.next_bias,
        exact.next_bias,
        rtol=0,
        atol=2 * histogram.bin_width,
    )


def test_histogram_error_decreases_with_more_bins():
    scores, cutoff, old_bias = scores_and_cutoffs(tokens=128)
    exact = ExactQuantileBalancer(4, 2).compute_next_bias(
        scores, cutoff, old_bias
    ).next_bias
    errors = []
    for bins in (8, 512):
        histogram = HistogramQuantileBalancer(4, 2, bins, -1.0, 1.0)
        histogram.accumulate(scores, cutoff)
        approximate = histogram.compute_next_bias(old_bias).next_bias
        errors.append((approximate - exact).abs().max())
    assert errors[1] < errors[0]


def test_histogram_microbatch_accumulation_matches_single_batch_and_reset():
    scores, cutoff, old_bias = scores_and_cutoffs()
    single = HistogramQuantileBalancer(4, 2, 64, -1.0, 1.0)
    split = copy.deepcopy(single)
    single.accumulate(scores, cutoff)
    split.accumulate(scores[:11], cutoff[:11])
    split.accumulate(scores[11:], cutoff[11:])
    torch.testing.assert_close(single.counts, split.counts)
    torch.testing.assert_close(
        single.compute_next_bias(old_bias).next_bias,
        split.compute_next_bias(old_bias).next_bias,
    )
    split.reset()
    assert torch.count_nonzero(split.counts) == 0
    assert torch.count_nonzero(split.underflow) == 0
    assert torch.count_nonzero(split.overflow) == 0
    assert not hasattr(split, "margins")


def test_histogram_accounts_for_underflow_and_overflow_in_edge_bins():
    histogram = HistogramQuantileBalancer(4, 2, 8, -0.5, 0.5)
    scores = torch.tensor(
        [[-2.0, 0.0, 2.0, 0.25], [-3.0, 1.0, 3.0, -0.25]]
    )
    cutoff = torch.zeros(2)
    histogram.accumulate(scores, cutoff)
    assert histogram.underflow.tolist() == [2, 0, 0, 0]
    assert histogram.overflow.tolist() == [0, 1, 2, 0]
    assert histogram.counts[:, 0].sum() >= 2
    assert histogram.counts[:, -1].sum() >= 3


def test_qb_update_is_causal_changes_only_future_routing_and_is_frozen_in_eval():
    baseline = tiny_moe().double().train()
    updating = copy.deepcopy(baseline)
    x = torch.randn(4, 3, 8, dtype=torch.float64)
    expected = baseline(x)
    actual = updating(x, update_routing_bias=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(updating.routing_bias, baseline.routing_bias)
    debug_updating = copy.deepcopy(baseline)
    enriched = debug_updating(
        x,
        update_routing_bias=True,
        return_router_diagnostics=True,
    )
    assert enriched.router_output.raw_scores is None
    assert enriched.router_output.biased_scores is None
    next_baseline = baseline(
        x, return_router_diagnostics=True
    ).router_output.selected_experts
    next_updated = updating(
        x, return_router_diagnostics=True
    ).router_output.selected_experts
    assert not torch.equal(next_baseline, next_updated)
    updating.eval()
    snapshot = updating.routing_bias.clone()
    updating(x)
    torch.testing.assert_close(updating.routing_bias, snapshot)
    with pytest.raises(RuntimeError):
        updating(x, update_routing_bias=True)


def test_histogram_logical_batch_accumulates_without_intermediate_commit():
    model = tiny_moe(
        quantile_backend="histogram",
        histogram_num_bins=64,
        histogram_min_margin=-1.0,
        histogram_max_margin=1.0,
    ).train()
    model.begin_balance_accumulation()
    old_bias = model.routing_bias.clone()
    model(torch.randn(2, 3, 8))
    torch.testing.assert_close(model.routing_bias, old_bias)
    first_counts = model.histogram_balancer.counts.clone()
    model(torch.randn(1, 4, 8))
    assert torch.all(
        model.histogram_balancer.counts.sum(-1)
        == first_counts.sum(-1) + 4
    )
    update = model.finalize_and_commit_balance()
    torch.testing.assert_close(model.routing_bias, update.next_bias)
    assert torch.count_nonzero(model.histogram_balancer.counts) == 0
