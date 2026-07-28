import copy

import pytest
import torch
import torch.nn.functional as F

from src.vision import VisionPatchEmbedding


@pytest.mark.parametrize(
    "shape,patch,expected",
    [((2, 3, 28, 42), 14, (2, 6, 12)), ((1, 1, 12, 20), (3, 5), (1, 16, 8))],
)
def test_patch_embedding_rectangular_shapes(shape, patch, expected):
    module = VisionPatchEmbedding(shape[1], expected[-1], patch)
    tokens, grid = module(torch.randn(shape))
    assert tokens.shape == expected
    assert grid[0] * grid[1] == expected[1]


def test_patch_embedding_matches_conv2d_equation_exactly():
    module = VisionPatchEmbedding(1, 2, 2, bias=True)
    with torch.no_grad():
        module.projection.weight.copy_(
            torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
        )
        module.projection.bias.copy_(torch.tensor([1.0, -2.0]))
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    actual, grid = module(image)
    expected = F.conv2d(
        image, module.projection.weight, module.projection.bias, stride=2
    ).flatten(2).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert grid == (2, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(in_channels=0, embed_dim=8, patch_size=2),
        dict(in_channels=3, embed_dim=0, patch_size=2),
        dict(in_channels=3, embed_dim=8, patch_size=0),
        dict(in_channels=3, embed_dim=8, patch_size=(2, 0)),
    ],
)
def test_patch_embedding_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        VisionPatchEmbedding(**kwargs)


@pytest.mark.parametrize("shape", [(3, 28, 28), (2, 3, 28), (2, 3, 28, 28, 1)])
def test_patch_embedding_rejects_wrong_rank(shape):
    module = VisionPatchEmbedding(3, 8, 14)
    with pytest.raises(ValueError, match="\\[B, C, H, W\\]"):
        module(torch.randn(shape))


def test_patch_embedding_rejects_channel_mismatch():
    with pytest.raises(ValueError, match="3 input channels"):
        VisionPatchEmbedding(3, 8, 14)(torch.randn(2, 1, 28, 28))


@pytest.mark.parametrize("shape", [(1, 3, 29, 28), (1, 3, 28, 43)])
def test_patch_embedding_rejects_non_divisible_images(shape):
    with pytest.raises(ValueError, match="must be divisible"):
        VisionPatchEmbedding(3, 8, 14)(torch.randn(shape))


def test_patch_embedding_bias_policy_is_structural():
    assert VisionPatchEmbedding(3, 8, 2, bias=False).projection.bias is None
    assert VisionPatchEmbedding(3, 8, 2, bias=True).projection.bias is not None


def test_patch_embedding_backward_reaches_input_and_weight():
    module = VisionPatchEmbedding(3, 8, 2)
    image = torch.randn(2, 3, 4, 6, requires_grad=True)
    output, _ = module(image)
    output.square().mean().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert module.projection.weight.grad is not None
    assert torch.isfinite(module.projection.weight.grad).all()


def test_patch_embedding_state_dict_roundtrip_is_exact():
    module = VisionPatchEmbedding(3, 8, 2, bias=True)
    clone = copy.deepcopy(module)
    clone.load_state_dict(module.state_dict())
    image = torch.randn(2, 3, 4, 6)
    torch.testing.assert_close(module(image)[0], clone(image)[0], rtol=0, atol=0)


def test_patch_embedding_bfloat16_preserves_dtype():
    module = VisionPatchEmbedding(3, 8, 2).to(torch.bfloat16)
    output, _ = module(torch.randn(2, 3, 4, 4, dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16

