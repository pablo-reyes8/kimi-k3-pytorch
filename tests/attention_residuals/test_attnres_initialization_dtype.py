import pytest
import torch

from src.attention_residuals import AttentionResidualSite
from tests.attention_residuals.conftest import attnres_backbone


@pytest.mark.parametrize("depth_mode", ["full", "block"])
def test_all_queries_including_final_are_exactly_zero_initialized(depth_mode):
    model = attnres_backbone(depth_mode=depth_mode)
    queries = [
        site.pseudo_query
        for layer in model.layers
        for site in (
            layer.pre_attention_attnres,
            layer.pre_ffn_attnres,
        )
    ] + [model.final_output_attnres.pseudo_query]
    assert len(queries) == 2 * len(model.layers) + 1
    assert all(torch.count_nonzero(query) == 0 for query in queries)


def test_every_site_owns_independent_query_and_key_norm():
    model = attnres_backbone(depth_mode="block")
    sites = [
        site
        for layer in model.layers
        for site in (
            layer.pre_attention_attnres,
            layer.pre_ffn_attnres,
        )
    ] + [model.final_output_attnres]
    query_ptrs = [site.pseudo_query.data_ptr() for site in sites]
    norm_ptrs = [site.key_norm.weight.data_ptr() for site in sites]
    assert len(query_ptrs) == len(set(query_ptrs))
    assert len(norm_ptrs) == len(set(norm_ptrs))


def test_bfloat16_uses_fp32_logits_and_returns_value_dtype():
    site = AttentionResidualSite(4).bfloat16()
    sources = torch.randn(2, 3, 5, 4).bfloat16()
    output = site(sources, return_weights=True, return_logits=True)
    assert output.logits.dtype == torch.float32
    assert output.weights.dtype == torch.float32
    assert output.mixed_state.dtype == torch.bfloat16
    assert torch.isfinite(output.mixed_state.float()).all()


def test_disabling_precision_policy_keeps_bfloat16_computation():
    site = AttentionResidualSite(
        4, logits_in_fp32=False, weighted_sum_in_fp32=False
    ).bfloat16()
    output = site(
        torch.randn(1, 2, 3, 4).bfloat16(),
        return_weights=True,
        return_logits=True,
    )
    assert output.logits.dtype == torch.bfloat16
    assert output.weights.dtype == torch.bfloat16
    assert output.mixed_state.dtype == torch.bfloat16


def test_float64_precision_is_preserved():
    site = AttentionResidualSite(4).double()
    output = site(
        torch.randn(1, 2, 3, 4, dtype=torch.float64),
        return_weights=True,
        return_logits=True,
    )
    assert output.logits.dtype == torch.float64
    assert output.weights.dtype == torch.float64
    assert output.mixed_state.dtype == torch.float64


def test_extreme_logits_remain_finite_without_temperature():
    site = AttentionResidualSite(4).double()
    with torch.no_grad():
        site.pseudo_query.fill_(1e10)
    output = site(
        torch.randn(2, 3, 20, 4, dtype=torch.float64) * 1e10,
        return_weights=True,
    )
    assert torch.isfinite(output.mixed_state).all()
    assert torch.isfinite(output.weights).all()
    torch.testing.assert_close(
        output.weights.sum(-1), torch.ones_like(output.weights[..., 0])
    )
