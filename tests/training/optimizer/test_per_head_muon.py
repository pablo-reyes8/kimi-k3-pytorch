import pytest
import torch

from training.optimizer import (
    HeadMatrixLayout,
    merge_head_matrix,
    per_head_orthogonalize,
    split_head_matrix,
    zeropower_via_newton_schulz,
)
from training.optimizer.newton_schulz import match_update_rms


@pytest.mark.parametrize("axis,shape", [(0, (6, 4)), (1, (4, 6))])
def test_split_merge_is_identity_and_respects_head_axis(axis, shape):
    matrix = torch.arange(24.0).reshape(shape)
    layout = HeadMatrixLayout(
        num_heads=3, head_dim=2, head_axis=axis,
        input_dim=shape[1], output_dim=shape[0],
    )
    blocks = split_head_matrix(matrix, layout)
    assert len(blocks) == 3
    assert torch.equal(merge_head_matrix(blocks, layout), matrix)


def test_per_head_muon_matches_explicit_independent_head_loop():
    torch.manual_seed(1)
    matrix = torch.randn(6, 5)
    layout = HeadMatrixLayout(3, 2, 0, 5, 6)
    actual, metrics = per_head_orthogonalize(
        matrix, layout, steps=4, eps=1e-7, rms_scaling=True
    )
    expected_blocks = []
    for block in split_head_matrix(matrix, layout):
        expected_blocks.append(match_update_rms(
            zeropower_via_newton_schulz(block, steps=4),
            mode="shape",
        ))
    torch.testing.assert_close(
        actual, merge_head_matrix(expected_blocks, layout)
    )
    assert all(isinstance(value, float) for value in metrics.values())
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_per_head_balances_heads_with_very_different_raw_scales():
    small = torch.randn(2, 4) * 1e-4
    large = torch.randn(2, 4) * 1e4
    matrix = torch.cat((small, large))
    layout = HeadMatrixLayout(2, 2, 0, 4, 4)
    update, metrics = per_head_orthogonalize(
        matrix, layout, steps=5, eps=1e-7, rms_scaling=True
    )
    head_rms = torch.tensor([
        block.square().mean().sqrt()
        for block in split_head_matrix(update, layout)
    ])
    assert head_rms.max() / head_rms.min() < 1.1
    assert metrics["per_head_muon/head_update_rms_cv"] < 0.05


def test_every_per_head_summary_metric_has_its_stated_semantics():
    matrix = torch.cat((torch.ones(2, 3), torch.full((2, 3), 2.0)))
    layout = HeadMatrixLayout(2, 2, 0, 3, 4)
    update, metrics = per_head_orthogonalize(
        matrix, layout, steps=3, eps=1e-7, rms_scaling=False
    )
    expected_keys = {
        f"per_head_muon/{family}_{stat}"
        for family in (
            "raw_momentum_rms", "orthogonal_update_rms",
            "head_update_rms",
        )
        for stat in ("mean", "std", "min", "max", "median", "cv")
    } | {
        "per_head_muon/head_update_rms_max_to_median",
        "per_head_muon/fraction_heads_near_zero_update",
    }
    assert metrics.keys() == expected_keys
    assert metrics["per_head_muon/raw_momentum_rms_mean"] == pytest.approx(1.5)
    assert metrics["per_head_muon/raw_momentum_rms_std"] == pytest.approx(0.5)
    assert metrics["per_head_muon/raw_momentum_rms_min"] == pytest.approx(1.0)
    assert metrics["per_head_muon/raw_momentum_rms_max"] == pytest.approx(2.0)
    assert metrics["per_head_muon/raw_momentum_rms_median"] == pytest.approx(1.0)
    assert metrics["per_head_muon/raw_momentum_rms_cv"] == pytest.approx(1 / 3)
    final = torch.tensor([
        block.square().mean().sqrt()
        for block in split_head_matrix(update, layout)
    ])
    assert metrics["per_head_muon/head_update_rms_mean"] == pytest.approx(
        final.mean().item()
    )
    assert metrics["per_head_muon/head_update_rms_std"] == pytest.approx(
        final.std(unbiased=False).item()
    )


def test_fused_qkv_slices_are_processed_without_mixing():
    matrix = torch.tensor([
        [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [1.0, 2.0, 0.0], [0.0, 1.0, 3.0],
    ])
    layout = HeadMatrixLayout(
        num_heads=1, head_dim=2, head_axis=0,
        input_dim=3, output_dim=6, packed_kind="fused_qkv",
        qkv_slices=(slice(0, 2), slice(2, 4), slice(4, 6)),
    )
    update, metrics = per_head_orthogonalize(
        matrix, layout, steps=3, eps=1e-7
    )
    assert update.shape == matrix.shape
    for packed_slice in layout.qkv_slices:
        child = HeadMatrixLayout(1, 2, 0, 3, 2)
        expected, _ = per_head_orthogonalize(
            matrix[packed_slice], child, steps=3, eps=1e-7
        )
        torch.testing.assert_close(update[packed_slice], expected)
    assert not torch.equal(update[0:2], update[2:4])
    assert metrics["per_head_muon/fraction_heads_near_zero_update"] == 0
