import copy

import pytest
import torch
import torch.nn as nn

from src.vision import (
    HierarchicalMoonViTEncoder,
    HierarchicalTokenPool,
    HierarchicalVisionConfig,
)
from src.transformer_modules.rms_norm import RMSNorm
from tests.MoonViT.helpers import tiny_hierarchical


@pytest.mark.parametrize(
    "grid,expected",
    [((4, 6), (2, 3)), ((3, 5), (2, 3)), ((1, 1), (1, 1))],
)
def test_hierarchical_pool_shape_equation(grid, expected):
    pool = HierarchicalTokenPool(8, 12)
    output, new_grid, _ = pool(torch.randn(2, grid[0] * grid[1], 8), grid)
    assert output.shape == (2, expected[0] * expected[1], 12)
    assert new_grid == expected


def test_hierarchical_pool_matches_explicit_convolution_equation():
    pool = HierarchicalTokenPool(2, 3, norm_type="layernorm", bias=True)
    tokens = torch.randn(1, 12, 2)
    spatial = tokens.reshape(1, 3, 4, 2).permute(0, 3, 1, 2)
    expected = pool.projection(pool.depthwise(spatial))
    expected = expected.flatten(2).transpose(1, 2)
    expected = pool.norm(expected)
    actual, grid, _ = pool(tokens, (3, 4))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert grid == (2, 2)


def test_hierarchical_pool_propagates_valid_mask_by_any_receptive_token():
    pool = HierarchicalTokenPool(4, 8)
    tokens = torch.randn(1, 16, 4)
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[0, 0] = True
    _, grid, pooled_mask = pool(tokens, (4, 4), mask)
    assert grid == (2, 2)
    expected = torch.tensor([[True, False, False, False]])
    torch.testing.assert_close(pooled_mask, expected)


def test_hierarchical_pool_bias_policy_and_parameter_shapes():
    no_bias = HierarchicalTokenPool(8, 12, bias=False)
    assert no_bias.depthwise.bias is None and no_bias.projection.bias is None
    assert no_bias.depthwise.weight.shape == (8, 1, 3, 3)
    assert no_bias.projection.weight.shape == (12, 8, 1, 1)
    biased = HierarchicalTokenPool(8, 12, bias=True)
    assert biased.depthwise.bias is not None and biased.projection.bias is not None


def test_hierarchical_pool_validates_grid_width_and_mask():
    pool = HierarchicalTokenPool(8, 12)
    with pytest.raises(ValueError):
        pool(torch.randn(2, 5, 8), (2, 3))
    with pytest.raises(ValueError):
        pool(torch.randn(2, 6, 7), (2, 3))
    with pytest.raises(ValueError):
        pool(torch.randn(2, 6, 8), (2, 3), torch.ones(2, 5, dtype=torch.bool))


def test_hierarchical_pool_backward_reaches_all_inputs_and_parameters():
    pool = HierarchicalTokenPool(8, 12)
    tokens = torch.randn(2, 12, 8, requires_grad=True)
    pool(tokens, (3, 4))[0].square().mean().backward()
    assert tokens.grad is not None
    assert all(parameter.grad is not None for parameter in pool.parameters())


def test_hierarchical_encoder_stage_shapes_and_diagnostics():
    model = tiny_hierarchical(depths=(2, 1))
    output = model(
        torch.randn(2, 3, 56, 84),
        output_hidden_states=True,
        output_attentions=True,
    )
    assert output.grid_size == (2, 3)
    assert output.last_hidden_state.shape == (2, 6, 48)
    assert len(output.attentions) == 3
    assert [weights.shape[-1] for weights in output.attentions] == [24, 24, 6]
    assert output.hidden_states[-1].shape == output.last_hidden_state.shape


def test_hierarchical_encoder_supports_dynamic_rectangular_resolution():
    model = tiny_hierarchical()
    output = model(torch.randn(1, 3, 84, 56))
    assert output.grid_size == (3, 2)
    assert output.last_hidden_state.shape == (1, 6, 48)


def test_hierarchical_encoder_is_global_not_windowed():
    model = tiny_hierarchical(depths=(1, 1)).eval()
    output = model(
        torch.randn(1, 3, 56, 84), output_attentions=True
    )
    first_attention = output.attentions[0]
    assert first_attention.shape[-2:] == (24, 24)
    assert torch.all(first_attention > 0)


def test_hierarchical_encoder_default_policy_is_bias_free_rmsnorm():
    model = tiny_hierarchical()
    assert not any(name.endswith(".bias") for name, _ in model.named_parameters())
    assert any(isinstance(module, RMSNorm) for module in model.modules())
    assert not any(
        isinstance(module, nn.Linear) and module.bias is not None
        for module in model.modules()
    )


def test_hierarchical_encoder_mask_reaches_each_stage():
    model = tiny_hierarchical(depths=(1, 1))
    mask = torch.ones(1, 24, dtype=torch.bool)
    mask[:, -6:] = False
    output = model(
        torch.randn(1, 3, 56, 84), mask, output_attentions=True
    )
    assert torch.count_nonzero(output.attentions[0][..., -6:]) == 0
    assert output.attentions[1].shape[-1] == 6


def test_hierarchical_encoder_backward_and_state_roundtrip():
    model = tiny_hierarchical()
    image = torch.randn(2, 3, 56, 84, requires_grad=True)
    model(image).last_hidden_state.square().mean().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
    clone = copy.deepcopy(model).eval()
    model.eval()
    reference = torch.randn(1, 3, 56, 84)
    torch.testing.assert_close(
        model(reference).last_hidden_state,
        clone(reference).last_hidden_state,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embed_dims": ()},
        {"embed_dims": (24, 48), "depths": (1,), "num_heads": (6, 6)},
        {"embed_dims": (25,), "depths": (1,), "num_heads": (6,)},
        {"depths": (0,)},
        {"position_embedding_type": "relative"},
        {"dropout": 1},
    ],
)
def test_hierarchical_config_validation(kwargs):
    base = dict(embed_dims=(24,), depths=(1,), num_heads=(6,))
    base.update(kwargs)
    with pytest.raises(ValueError):
        HierarchicalVisionConfig(**base)


def test_hierarchical_encoder_bfloat16():
    model = tiny_hierarchical().to(torch.bfloat16)
    image = torch.randn(
        1, 3, 56, 84, dtype=torch.bfloat16, requires_grad=True
    )
    output = model(image).last_hidden_state
    assert output.dtype == torch.bfloat16
    output.float().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()

