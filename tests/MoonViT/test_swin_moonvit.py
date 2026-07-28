import copy

import pytest
import torch
import torch.nn as nn

from src.vision import (
    SwinMoonViTEncoder,
    SwinPatchMerging,
    SwinTransformerBlock,
    SwinVisionConfig,
    WindowSelfAttention,
)
from src.vision.swin_encoder import _partition_windows, _reverse_windows
from tests.MoonViT.helpers import tiny_swin


@pytest.mark.parametrize("shape,window", [((2, 4, 6, 3), 2), ((1, 3, 5, 4), 4)])
def test_window_partition_reverse_is_exact_even_with_padding(shape, window):
    x = torch.randn(shape)
    windows, metadata = _partition_windows(x, window)
    restored = _reverse_windows(windows, window, metadata)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)


def test_window_attention_matches_uniform_average_when_qk_are_zero():
    attention = WindowSelfAttention(
        4, 2, 2, use_relative_position_bias=False, proj_bias=False
    )
    with torch.no_grad():
        attention.qkv.weight.zero_()
        attention.qkv.weight[8:12].copy_(torch.eye(4))
        attention.projection.weight.copy_(torch.eye(4))
    windows = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    output, weights = attention(windows, output_attentions=True)
    expected = windows.mean(dim=1, keepdim=True).expand_as(windows)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weights, torch.full_like(weights, 0.25))


def test_window_attention_relative_index_covers_full_table():
    attention = WindowSelfAttention(8, 2, 3)
    assert attention.relative_position_bias.shape == (25, 2)
    assert attention.relative_position_index.min() == 0
    assert attention.relative_position_index.max() == 24
    assert attention.relative_position_index.shape == (9, 9)


def test_window_attention_mask_has_exact_zeros_and_normalized_rows():
    attention = WindowSelfAttention(8, 2, 2)
    mask = torch.eye(4, dtype=torch.bool).unsqueeze(0)
    _, weights = attention(
        torch.randn(1, 4, 8), mask, output_attentions=True
    )
    expected = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 2, 4, 4)
    torch.testing.assert_close(weights, expected)


def test_window_attention_backward_reaches_relative_bias():
    attention = WindowSelfAttention(8, 2, 2)
    windows = torch.randn(3, 4, 8, requires_grad=True)
    attention(windows)[0].square().mean().backward()
    assert windows.grad is not None
    assert all(parameter.grad is not None for parameter in attention.parameters())


def test_swin_patch_merging_exact_tl_tr_bl_br_order_before_projection():
    merger = SwinPatchMerging(
        1, 4, norm_type="layernorm", norm_eps=1e-6, bias=False
    )
    merger.norm = nn.Identity()
    with torch.no_grad():
        merger.reduction.weight.copy_(torch.eye(4))
    tokens = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
    output, grid, _ = merger(tokens, (4, 4))
    expected = torch.tensor(
        [[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert grid == (2, 2)


def test_swin_patch_merging_pads_odd_grid_and_propagates_mask():
    merger = SwinPatchMerging(
        4, 8, norm_type="rmsnorm", norm_eps=1e-6, bias=False
    )
    mask = torch.zeros(1, 15, dtype=torch.bool)
    mask[0, -1] = True
    output, grid, pooled_mask = merger(torch.randn(1, 15, 4), (3, 5), mask)
    assert output.shape == (1, 6, 8)
    assert grid == (2, 3)
    assert pooled_mask.sum() == 1 and pooled_mask[0, -1]


def test_regular_and_shifted_blocks_preserve_shape_but_use_different_masks():
    kwargs = dict(
        dim=8, num_heads=2, window_size=2, mlp_ratio=2,
        norm_type="rmsnorm", norm_eps=1e-6, qkv_bias=False,
        proj_bias=False, mlp_bias=False, dropout=0,
        attention_dropout=0, drop_path=0, use_relative_position_bias=True,
    )
    regular = SwinTransformerBlock(shift_size=0, **kwargs)
    shifted = SwinTransformerBlock(shift_size=1, **kwargs)
    shifted.load_state_dict(regular.state_dict())
    tokens = torch.randn(1, 16, 8)
    regular_output, regular_weights = regular(
        tokens, (4, 4), output_attentions=True
    )
    shifted_output, shifted_weights = shifted(
        tokens, (4, 4), output_attentions=True
    )
    assert regular_output.shape == shifted_output.shape == tokens.shape
    assert not torch.allclose(regular_output, shifted_output)
    assert torch.count_nonzero(shifted_weights == 0) > torch.count_nonzero(
        regular_weights == 0
    )


def test_shifted_block_roundtrip_is_identity_when_branches_zero():
    block = SwinTransformerBlock(
        8, 2, 2, 1, mlp_ratio=2, norm_type="rmsnorm", norm_eps=1e-6,
        qkv_bias=False, proj_bias=False, mlp_bias=False, dropout=0,
        attention_dropout=0, drop_path=0, use_relative_position_bias=True,
    )
    with torch.no_grad():
        for parameter in block.attention.parameters():
            parameter.zero_()
        for parameter in block.mlp.parameters():
            parameter.zero_()
    tokens = torch.randn(2, 15, 8)
    torch.testing.assert_close(
        block(tokens, (3, 5))[0], tokens, rtol=0, atol=0
    )


def test_swin_encoder_shapes_diagnostics_and_stage_merging():
    model = tiny_swin()
    output = model(
        torch.randn(2, 3, 56, 84),
        output_hidden_states=True,
        output_attentions=True,
    )
    assert output.grid_size == (2, 3)
    assert output.last_hidden_state.shape == (2, 6, 48)
    assert len(output.attentions) == 4
    assert len(output.hidden_states) == 6
    assert output.hidden_states[-1].shape == output.last_hidden_state.shape


def test_swin_encoder_supports_dynamic_rectangular_and_odd_patch_grids():
    model = tiny_swin()
    output = model(torch.randn(1, 3, 42, 70))
    assert output.grid_size == (2, 3)
    assert output.last_hidden_state.shape == (1, 6, 48)


def test_swin_encoder_is_windowed_not_global():
    model = tiny_swin(depths=(1, 1))
    output = model(torch.randn(1, 3, 56, 84), output_attentions=True)
    assert output.attentions[0].shape[-1] == model.config.window_size**2
    assert output.attentions[0].shape[-1] < 24


def test_swin_encoder_default_linear_policy_is_bias_free():
    model = tiny_swin()
    linear_biases = [
        module.bias for module in model.modules() if isinstance(module, nn.Linear)
    ]
    assert all(bias is None for bias in linear_biases)
    assert model.patch_embedding.projection.bias is None


def test_swin_encoder_backward_and_state_roundtrip():
    model = tiny_swin()
    image = torch.randn(2, 3, 56, 84, requires_grad=True)
    model(image).last_hidden_state.square().mean().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
    model.eval()
    clone = copy.deepcopy(model).eval()
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
        {"window_size": 0},
        {"drop_path_rate": 1},
    ],
)
def test_swin_config_validation(kwargs):
    base = dict(embed_dims=(24,), depths=(1,), num_heads=(6,))
    base.update(kwargs)
    with pytest.raises(ValueError):
        SwinVisionConfig(**base)


def test_swin_encoder_bfloat16_forward_backward():
    model = tiny_swin().to(torch.bfloat16)
    image = torch.randn(
        1, 3, 56, 84, dtype=torch.bfloat16, requires_grad=True
    )
    output = model(image).last_hidden_state
    assert output.dtype == torch.bfloat16
    output.float().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()

