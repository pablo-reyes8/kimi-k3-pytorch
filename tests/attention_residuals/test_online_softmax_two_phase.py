import copy

import pytest
import torch

from src.attention_residuals import (
    AttentionResidualSite,
    BlockAttentionResidualController,
    DepthSiteMetadata,
    depth_softmax_stats,
    merge_depth_softmax_stats,
    normalize_depth_softmax_stats,
    precompute_inter_block_stats,
    single_source_stats,
    weights_from_stats,
)


def make_sites(count, d_model=5, block_size=4):
    sites = []
    for index in range(count):
        site = AttentionResidualSite(
            d_model,
            metadata=DepthSiteMetadata(
                index,
                index // 2,
                "pre_attention" if index % 2 == 0 else "pre_ffn",
                "kda",
                0,
                index // 2,
                index // block_size,
                index % block_size,
            ),
        ).double()
        with torch.no_grad():
            site.pseudo_query.normal_()
            site.key_norm.weight.uniform_(0.5, 1.5)
        sites.append(site)
    return tuple(sites)


def test_online_merge_matches_concatenate_softmax_reference():
    logits = torch.randn(2, 3, 7, dtype=torch.float64) * 10
    values = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    first = depth_softmax_stats(logits[:, :, :4], values[:, :, :4])
    second = depth_softmax_stats(logits[:, :, 4:], values[:, :, 4:])
    merged = merge_depth_softmax_stats(first, second)
    actual = normalize_depth_softmax_stats(merged, torch.float64)
    expected = torch.einsum(
        "bts,btsd->btd", torch.softmax(logits, -1), values
    )
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(
        weights_from_stats(merged), torch.softmax(logits, -1),
        rtol=2e-14, atol=2e-14,
    )


def test_single_source_stats_uses_unit_denominator_and_raw_value():
    logit = torch.randn(2, 3, dtype=torch.float64)
    value = torch.randn(2, 3, 5, dtype=torch.float64)
    stats = single_source_stats(logit, value)
    torch.testing.assert_close(stats.max_logit, logit)
    torch.testing.assert_close(stats.exp_sum, torch.ones_like(logit))
    torch.testing.assert_close(stats.weighted_sum, value)


def test_online_merge_extreme_logits_is_finite():
    first = depth_softmax_stats(
        torch.tensor([[[1e10, -1e10]]], dtype=torch.float64),
        torch.tensor([[[[3.0], [5.0]]]], dtype=torch.float64),
    )
    second = single_source_stats(
        torch.tensor([[1e10 - 1]], dtype=torch.float64),
        torch.tensor([[[9.0]]], dtype=torch.float64),
    )
    output = normalize_depth_softmax_stats(
        merge_depth_softmax_stats(first, second), torch.float64
    )
    assert torch.isfinite(output).all()


def simulate(controller, sites, embedding):
    state = controller.initialize(embedding)
    mixed_states, weights = [], []
    by_block = {}
    for site in sites:
        by_block.setdefault(site.metadata.depth_block_index, []).append(site)
    by_block = {key: tuple(value) for key, value in by_block.items()}
    for site in sites:
        controller.prepare_depth_block(
            state, by_block[site.metadata.depth_block_index]
        )
        mixed = controller.mix_for_site(
            site, state, return_weights=True
        )
        mixed_states.append(mixed.mixed_state)
        weights.append(mixed.weights)
        controller.append_output(
            state, torch.tanh(mixed.mixed_state * (site.metadata.site_index + 1))
        )
    final = AttentionResidualSite(
        embedding.shape[-1],
        metadata=DepthSiteMetadata(
            len(sites), None, "final_output", None, None, None, None, None
        ),
    ).double()
    return mixed_states, weights, controller.finalize(
        state, final, return_weights=True
    ), state


def test_eager_and_two_phase_outputs_weights_and_scan_count():
    sites = make_sites(10)
    eager_sites = copy.deepcopy(sites)
    embedding = torch.randn(2, 3, 5, dtype=torch.float64)
    eager = simulate(
        BlockAttentionResidualController(4, "eager"),
        eager_sites,
        embedding,
    )
    two_phase = simulate(
        BlockAttentionResidualController(4, "two_phase"),
        sites,
        embedding,
    )
    for left, right in zip(eager[0], two_phase[0]):
        torch.testing.assert_close(left, right, rtol=2e-14, atol=2e-14)
    for left, right in zip(eager[1], two_phase[1]):
        torch.testing.assert_close(left, right, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(
        eager[2].mixed_state, two_phase[2].mixed_state,
        rtol=2e-14, atol=2e-14,
    )
    assert eager[3].inter_block_scan_count == 0
    assert two_phase[3].inter_block_scan_count == 3


def test_vectorized_phase_one_matches_each_site_eager_logits():
    sites = make_sites(5)
    sources = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    stats = precompute_inter_block_stats(sources, sites)
    for site in sites:
        eager = site(sources, return_logits=True)
        torch.testing.assert_close(
            stats[site.metadata.site_index].logits,
            eager.logits,
            rtol=2e-14,
            atol=2e-14,
        )


def test_two_phase_honors_independent_bfloat16_dtype_policies():
    site = AttentionResidualSite(
        4,
        logits_in_fp32=False,
        weighted_sum_in_fp32=True,
        metadata=DepthSiteMetadata(
            0, 0, "pre_attention", "kda", 0, 0, 0, 0
        ),
    ).bfloat16()
    with torch.no_grad():
        site.pseudo_query.normal_()
    sources = torch.randn(2, 3, 4, 4).bfloat16()
    eager = site(sources, return_logits=True)
    phase_one = precompute_inter_block_stats(sources, (site,))[0]
    actual = normalize_depth_softmax_stats(phase_one, torch.bfloat16)
    assert phase_one.logits.dtype == torch.bfloat16
    assert phase_one.weighted_sum.dtype == torch.float32
    torch.testing.assert_close(phase_one.logits, eager.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual, eager.mixed_state, rtol=3e-2, atol=8e-3)


def test_online_merge_gradcheck():
    logits_a = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
    values_a = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    logits_b = torch.randn(1, 2, 1, dtype=torch.float64, requires_grad=True)
    values_b = torch.randn(1, 2, 1, 3, dtype=torch.float64, requires_grad=True)

    def function(la, va, lb, vb):
        merged = merge_depth_softmax_stats(
            depth_softmax_stats(la, va),
            depth_softmax_stats(lb, vb),
        )
        return normalize_depth_softmax_stats(merged, torch.float64)

    assert torch.autograd.gradcheck(
        function, (logits_a, values_a, logits_b, values_b), fast_mode=True
    )
