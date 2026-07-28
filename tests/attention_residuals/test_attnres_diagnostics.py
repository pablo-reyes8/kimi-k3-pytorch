import math

import pytest
import torch

from src.attention_residuals import AttentionResidualSite, DepthSiteMetadata
from tests.attention_residuals.conftest import (
    activate_depth_queries,
    attnres_backbone,
)


def site_with_metadata(index=2):
    return AttentionResidualSite(
        4,
        metadata=DepthSiteMetadata(
            index, 1, "pre_attention", "kda", 0, 1, None, None
        ),
    ).double()


@pytest.mark.parametrize("sources", [1, 2, 5])
def test_uniform_query_diagnostic_entropy_and_weights(sources):
    site = site_with_metadata()
    output = site(
        torch.randn(2, 3, sources, 4, dtype=torch.float64),
        return_weights=True,
        return_stats=True,
    )
    stats = output.stats
    torch.testing.assert_close(
        output.weights.sum(-1),
        torch.ones_like(output.weights[..., 0]),
    )
    expected_entropy = 0.0 if sources == 1 else math.log(sources)
    torch.testing.assert_close(
        stats.weight_entropy,
        torch.tensor(expected_entropy),
        rtol=1e-6,
        atol=1e-7,
    )
    expected_normalized = 0.0 if sources == 1 else 1.0
    torch.testing.assert_close(
        stats.normalized_entropy,
        torch.tensor(expected_normalized),
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        stats.max_weight, torch.tensor(1 / sources), rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(stats.embedding_weight, stats.max_weight)


@pytest.mark.parametrize(
    "depth_mode,backend",
    [("full", "eager"), ("block", "eager"), ("block", "two_phase")],
)
def test_backbone_depth_outputs_matrix_mask_labels_and_global_stats(
    depth_mode, backend
):
    model = attnres_backbone(
        depth_mode=depth_mode, backend=backend, block_size=4
    ).eval()
    output = model(
        torch.randn(2, 5, 8),
        output_depth_weights=True,
        output_diagnostics=True,
    )
    depth = output.depth_outputs
    assert len(depth.site_stats) == 2 * len(model.layers)
    assert depth.final_output_stats.metadata.site_kind == "final_output"
    assert depth.averaged_weight_matrix.shape[0] == 2 * len(model.layers) + 1
    assert depth.source_mask.shape == depth.averaged_weight_matrix.shape
    for row, mask, stats, labels in zip(
        depth.averaged_weight_matrix,
        depth.source_mask,
        depth.site_stats + (depth.final_output_stats,),
        depth.source_labels,
    ):
        assert mask.sum() == stats.source_count
        torch.testing.assert_close(
            row[mask].sum(), torch.tensor(1.0), rtol=1e-6, atol=1e-6
        )
        assert len(labels) == stats.source_count
        assert labels[0] == "embedding"
    diagnostics = output.diagnostics
    for name in (
        "mean_embedding_weight",
        "mean_depth_entropy",
        "mean_retrieval_distance",
        "fraction_dominated_by_embedding",
        "fraction_dominated_by_most_recent",
    ):
        assert diagnostics[name].ndim == 0
        assert torch.isfinite(diagnostics[name])
    assert 0 <= diagnostics["fraction_dominated_by_embedding"] <= 1
    assert 0 <= diagnostics["fraction_dominated_by_most_recent"] <= 1


def test_block_partial_weight_exists_only_after_first_site_in_block():
    model = attnres_backbone(
        depth_mode="block", block_size=4
    ).eval()
    output = model(
        torch.randn(1, 3, 8), output_depth_weights=True
    )
    stats = output.depth_outputs.site_stats
    for index, item in enumerate(stats):
        if index % 4 == 0:
            assert item.current_partial_weight == 0
        else:
            assert item.current_partial_weight > 0
        assert item.number_of_completed_blocks == index // 4


def test_memory_accounting_full_vs_block():
    full = attnres_backbone(depth_mode="full").eval()(
        torch.randn(2, 5, 8), output_depth_weights=True
    ).depth_outputs
    block = attnres_backbone(
        depth_mode="block", block_size=4
    ).eval()(torch.randn(2, 5, 8), output_depth_weights=True).depth_outputs
    assert full.source_tensor_count == 11
    assert full.peak_source_count == 11
    assert full.source_elements == 11 * 2 * 5 * 8
    assert block.source_tensor_count == 4
    assert block.num_depth_blocks == 3
    assert block.partial_final_block_size == 2
    assert block.source_elements == 4 * 2 * 5 * 8
    assert block.source_elements < full.source_elements


def test_active_queries_produce_nonuniform_content_dependent_diagnostics():
    model = attnres_backbone(depth_mode="full").eval()
    activate_depth_queries(model)
    output = model(
        torch.randn(2, 5, 8), output_depth_weights=True
    )
    matrix = output.depth_outputs.averaged_weight_matrix
    mask = output.depth_outputs.source_mask
    assert any(
        row[valid].numel() > 1
        and not torch.allclose(
            row[valid], torch.full_like(row[valid], 1 / valid.sum())
        )
        for row, valid in zip(matrix, mask)
    )
