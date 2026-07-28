import pytest
import torch
import torch.nn.functional as F

from src.vision import (
    LearnedAbsolutePositionEmbedding,
    SpatialTokenPixelShuffle,
    VisionProjector,
)


def test_position_embedding_base_grid_returns_parameter_exactly():
    module = LearnedAbsolutePositionEmbedding((2, 3), 4)
    actual = module((2, 3))
    assert actual.data_ptr() == module.patch_positions.data_ptr()
    torch.testing.assert_close(actual, module.patch_positions, rtol=0, atol=0)


def test_position_interpolation_matches_explicit_bicubic_equation():
    module = LearnedAbsolutePositionEmbedding((2, 3), 4)
    actual = module((4, 5))
    expected = F.interpolate(
        module.patch_positions.reshape(1, 2, 3, 4).permute(0, 3, 1, 2),
        size=(4, 5),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).reshape(1, 20, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_position_cls_is_prepended_and_patch_positions_unchanged():
    module = LearnedAbsolutePositionEmbedding((2, 2), 4, use_cls_token=True)
    actual = module((2, 2))
    torch.testing.assert_close(actual[:, :1], module.cls_position, rtol=0, atol=0)
    torch.testing.assert_close(
        actual[:, 1:], module.patch_positions, rtol=0, atol=0
    )


def test_position_interpolation_gradients_reach_parameter():
    module = LearnedAbsolutePositionEmbedding((2, 3), 4)
    module((4, 5)).square().mean().backward()
    assert module.patch_positions.grad is not None
    assert module.patch_positions.grad.abs().sum() > 0


def test_position_requested_dtype_is_honored():
    module = LearnedAbsolutePositionEmbedding((2, 2), 4)
    assert module((3, 3), dtype=torch.bfloat16).dtype == torch.bfloat16


@pytest.mark.parametrize("grid", [(0, 2), (2, 0), (1, 2, 3)])
def test_position_rejects_invalid_grids(grid):
    with pytest.raises(ValueError):
        LearnedAbsolutePositionEmbedding((2, 2), 4)(grid)


def test_pixel_shuffle_exact_spatial_order():
    # Position values make TL/TR/BL/BR ordering unambiguous.
    tokens = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
    output = SpatialTokenPixelShuffle()(tokens, (4, 4))
    expected = torch.tensor(
        [[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(output.last_hidden_state, expected, rtol=0, atol=0)
    assert output.grid_size == (2, 2)


def test_pixel_shuffle_preserves_every_scalar_exactly():
    tokens = torch.randn(2, 24, 7)
    output = SpatialTokenPixelShuffle()(tokens, (4, 6)).last_hidden_state
    torch.testing.assert_close(
        output.flatten().sort().values,
        tokens.flatten().sort().values,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "shape,grid",
    [((2, 5, 4), (2, 3)), ((2, 6, 4), (3, 2)), ((2, 6, 4), (2, 3))],
)
def test_pixel_shuffle_rejects_count_or_odd_grid(shape, grid):
    with pytest.raises(ValueError):
        SpatialTokenPixelShuffle()(torch.randn(shape), grid)


def test_pixel_shuffle_backward_is_exact_unit_routing():
    tokens = torch.randn(2, 16, 3, requires_grad=True)
    SpatialTokenPixelShuffle()(tokens, (4, 4)).last_hidden_state.sum().backward()
    torch.testing.assert_close(tokens.grad, torch.ones_like(tokens))


def test_projector_matches_explicit_equation_and_bias_policy():
    projector = VisionProjector(8, 12, 6, bias=False)
    tokens = torch.randn(2, 3, 8)
    expected = projector.fc2(projector.activation(projector.fc1(tokens)))
    torch.testing.assert_close(projector(tokens), expected, rtol=0, atol=0)
    assert projector.fc1.bias is None and projector.fc2.bias is None


@pytest.mark.parametrize("activation", ["gelu", "silu"])
def test_projector_activation_variants_forward_backward(activation):
    projector = VisionProjector(8, 12, 6, activation=activation, bias=True)
    tokens = torch.randn(2, 3, 8, requires_grad=True)
    projector(tokens).square().mean().backward()
    assert tokens.grad is not None
    assert all(parameter.grad is not None for parameter in projector.parameters())


@pytest.mark.parametrize(
    "args",
    [(0, 4, 4), (4, 0, 4), (4, 4, 0), (4, 4, 4, "relu")],
)
def test_projector_rejects_invalid_configuration(args):
    with pytest.raises(ValueError):
        if len(args) == 3:
            VisionProjector(*args)
        else:
            VisionProjector(*args[:3], activation=args[3])


def test_projector_rejects_wrong_input_contract():
    projector = VisionProjector(8, 12, 6)
    with pytest.raises(ValueError):
        projector(torch.randn(2, 8))
    with pytest.raises(ValueError):
        projector(torch.randn(2, 3, 7))

