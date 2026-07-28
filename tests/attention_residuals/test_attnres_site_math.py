import math

import pytest
import torch

from src.attention_residuals import (
    AttentionResidualSite,
    DepthSiteMetadata,
    depth_softmax_mix_reference,
)


def metadata(index=0):
    return DepthSiteMetadata(
        index, 0, "pre_attention", "kda", 0, 0, None, None
    )


def test_single_source_is_identity_for_any_query():
    site = AttentionResidualSite(5, metadata=metadata()).double()
    with torch.no_grad():
        site.pseudo_query.normal_()
    sources = torch.randn(2, 3, 1, 5, dtype=torch.float64)
    output = site(sources, return_weights=True)
    torch.testing.assert_close(
        output.mixed_state, sources[:, :, 0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        output.weights, torch.ones(2, 3, 1, dtype=torch.float64)
    )


def test_zero_query_produces_uniform_weights_and_arithmetic_mean():
    site = AttentionResidualSite(7, metadata=metadata()).double()
    sources = torch.randn(2, 3, 5, 7, dtype=torch.float64)
    output = site(sources, return_weights=True)
    torch.testing.assert_close(
        output.weights, torch.full((2, 3, 5), 0.2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        output.mixed_state, sources.mean(dim=2), rtol=2e-15, atol=2e-15
    )
    assert not torch.equal(output.mixed_state, sources.sum(dim=2))


def test_manual_logits_weights_and_values_float64():
    site = AttentionResidualSite(4, eps=1e-8, metadata=metadata()).double()
    with torch.no_grad():
        site.pseudo_query.copy_(torch.tensor([0.5, -1.0, 2.0, 0.25]))
        site.key_norm.weight.copy_(torch.tensor([1.0, 1.5, 0.5, 2.0]))
    sources = torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4) - 8
    output = site(sources, return_weights=True, return_logits=True)
    normalized = sources * torch.rsqrt(
        sources.square().mean(-1, keepdim=True) + 1e-8
    )
    normalized = normalized * site.key_norm.weight
    logits = (normalized * site.pseudo_query).sum(-1)
    weights = torch.softmax(logits, -1)
    expected = (weights[..., None] * sources).sum(2)
    torch.testing.assert_close(output.logits, logits, rtol=0, atol=0)
    torch.testing.assert_close(output.weights, weights, rtol=0, atol=0)
    torch.testing.assert_close(output.mixed_state, expected, rtol=0, atol=0)


def test_softmax_axis_is_depth_when_all_axes_differ():
    site = AttentionResidualSite(5, metadata=metadata()).double()
    with torch.no_grad():
        site.pseudo_query.normal_()
    sources = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    weights = site(sources, return_weights=True).weights
    torch.testing.assert_close(
        weights.sum(dim=2), torch.ones(2, 3, dtype=torch.float64)
    )
    assert weights.shape == (2, 3, 4)


def test_keys_are_normalized_but_values_remain_raw():
    site = AttentionResidualSite(3, eps=1e-12, metadata=metadata()).double()
    with torch.no_grad():
        site.pseudo_query.copy_(torch.tensor([1.0, -0.5, 0.25]))
    source = torch.tensor([[[[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]]]])
    scaled = source.clone()
    scaled[:, :, 0] *= 10
    first = site(source.double(), return_weights=True)
    second = site(scaled.double(), return_weights=True)
    torch.testing.assert_close(first.weights, second.weights, rtol=1e-11, atol=1e-11)
    assert not torch.equal(first.mixed_state, second.mixed_state)


def test_no_hidden_sqrt_d_scaling():
    site = AttentionResidualSite(4, metadata=metadata()).double()
    with torch.no_grad():
        site.pseudo_query.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        )
    sources = torch.tensor(
        [[[[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]],
        dtype=torch.float64,
    )
    output = site(sources, return_weights=True, return_logits=True)
    scaled_weights = torch.softmax(output.logits / math.sqrt(4), dim=-1)
    assert not torch.allclose(output.weights, scaled_weights)


def test_source_permutation_changes_only_weight_order():
    site = AttentionResidualSite(4, metadata=metadata()).double()
    with torch.no_grad():
        site.pseudo_query.normal_()
    sources = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    baseline = site(sources, return_weights=True)
    permuted = site(sources[:, :, permutation], return_weights=True)
    torch.testing.assert_close(
        permuted.mixed_state, baseline.mixed_state, rtol=1e-14, atol=1e-14
    )
    torch.testing.assert_close(
        permuted.weights, baseline.weights[:, :, permutation]
    )


def test_token_and_batch_independence():
    site = AttentionResidualSite(4, metadata=metadata()).double()
    with torch.no_grad():
        site.pseudo_query.normal_()
    sources = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    baseline = site(sources).mixed_state
    changed = sources.clone()
    changed[1, 2] += 100
    actual = site(changed).mixed_state
    torch.testing.assert_close(actual[0], baseline[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1, :2], baseline[1, :2], rtol=0, atol=0)


def test_non_contiguous_sources_match_contiguous_reference():
    site = AttentionResidualSite(4, metadata=metadata()).double()
    with torch.no_grad():
        site.pseudo_query.normal_()
    sources = torch.randn(2, 5, 3, 4, dtype=torch.float64).transpose(1, 2)
    assert not sources.is_contiguous()
    actual = site(sources, return_weights=True)
    expected = site(sources.contiguous(), return_weights=True)
    torch.testing.assert_close(actual.mixed_state, expected.mixed_state)
    torch.testing.assert_close(actual.weights, expected.weights)


def test_static_query_still_produces_content_dependent_weights():
    site = AttentionResidualSite(4, metadata=metadata()).double()
    with torch.no_grad():
        site.pseudo_query.copy_(torch.tensor([1.0, 2.0, -1.0, 0.5]))
    sources = torch.randn(1, 2, 3, 4, dtype=torch.float64)
    weights = site(sources, return_weights=True).weights
    assert not torch.equal(weights[:, 0], weights[:, 1])


@pytest.mark.parametrize(
    "sources",
    [
        torch.randn(2, 3, 4),
        torch.randn(2, 3, 0, 4),
        torch.randn(2, 3, 4, 5),
    ],
)
def test_invalid_source_shapes_are_rejected(sources):
    with pytest.raises(ValueError):
        AttentionResidualSite(4)(sources)


def test_nan_query_and_nan_norm_are_rejected():
    site = AttentionResidualSite(4)
    sources = torch.randn(1, 1, 2, 4)
    with torch.no_grad():
        site.pseudo_query[0] = float("nan")
    with pytest.raises(ValueError):
        site(sources)
    with torch.no_grad():
        site.pseudo_query.zero_()
        site.key_norm.weight[0] = float("nan")
    with pytest.raises(ValueError):
        site(sources)


def test_reference_function_matches_site():
    site = AttentionResidualSite(4).double()
    with torch.no_grad():
        site.pseudo_query.normal_()
    sources = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    expected = depth_softmax_mix_reference(
        sources, site.pseudo_query, site.key_norm
    )
    actual = site(sources, return_weights=True, return_logits=True)
    for left, right in zip(
        (actual.mixed_state, actual.weights, actual.logits), expected
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
