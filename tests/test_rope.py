import pytest
import torch

from src.transformer_modules import RotaryEmbedding
from src.transformer_modules.rope_utils import rotate_half


def test_rotate_half_exact_values_shape_dtype_and_double_rotation():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    y = rotate_half(x)
    torch.testing.assert_close(y, expected)
    assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
    torch.testing.assert_close(rotate_half(y), -x)


def test_rotate_half_rejects_odd_dimension():
    with pytest.raises(ValueError, match="even"):
        rotate_half(torch.randn(2, 7))


def test_constructor_and_frequency_formula():
    rope = RotaryEmbedding(dim=16, rotary_dim=8, base=100.0)
    expected = 1.0 / (100.0 ** (torch.arange(0, 8, 2).float() / 8))
    assert rope.dim == 16 and rope.rotary_dim == 8 and rope.base == 100.0
    torch.testing.assert_close(rope.inv_freq, expected)
    assert "inv_freq" not in rope.state_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 0},
        {"dim": -1},
        {"dim": 8, "rotary_dim": 0},
        {"dim": 8, "rotary_dim": 10},
        {"dim": 8, "rotary_dim": 3},
        {"dim": 8, "base": 0},
    ],
)
def test_invalid_constructor_values_rejected(kwargs):
    with pytest.raises(ValueError):
        RotaryEmbedding(**kwargs)


def test_rotary_dim_defaults_to_full_dimension():
    assert RotaryEmbedding(16).rotary_dim == 16


@pytest.mark.parametrize("shape", [(2, 6, 3), (6, 16), (2, 6, 3, 16, 1)])
def test_non_four_dimensional_input_rejected(shape):
    with pytest.raises(ValueError):
        RotaryEmbedding(16)(torch.randn(*shape))


def test_wrong_head_dimension_rejected():
    with pytest.raises(ValueError, match="x.shape"):
        RotaryEmbedding(16)(torch.randn(2, 6, 3, 8))


def test_position_zero_identity_and_manual_equation():
    torch.manual_seed(2)
    rope = RotaryEmbedding(16, rotary_dim=8)
    x = torch.randn(2, 5, 3, 16)
    y = rope(x)
    torch.testing.assert_close(y[:, 0], x[:, 0])

    frequencies = torch.arange(5).float()[:, None] * rope.inv_freq[None]
    angles = torch.cat((frequencies, frequencies), dim=-1)
    expected_rotated = (
        x[..., 8:] * angles.cos()[None, :, None]
        + rotate_half(x[..., 8:]) * angles.sin()[None, :, None]
    )
    expected = torch.cat((x[..., :8], expected_rotated), dim=-1)
    torch.testing.assert_close(y, expected)


def test_partial_dimensions_unchanged_and_rotated_norm_preserved():
    rope = RotaryEmbedding(16, rotary_dim=8)
    x = torch.randn(2, 9, 3, 16)
    y = rope(x)
    torch.testing.assert_close(y[..., :8], x[..., :8], atol=0, rtol=0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(y[..., 8:], dim=-1),
        torch.linalg.vector_norm(x[..., 8:], dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_full_rope_preserves_dot_product_for_equal_positions():
    rope = RotaryEmbedding(16)
    q = torch.randn(2, 7, 3, 16)
    k = torch.randn(2, 7, 3, 16)
    q_rot, k_rot = rope(q), rope(k)
    original = (q * k).sum(-1)
    rotated = (q_rot * k_rot).sum(-1)
    torch.testing.assert_close(rotated, original, atol=2e-5, rtol=2e-5)


def test_start_position_equals_explicit_positions():
    rope = RotaryEmbedding(16)
    x = torch.randn(2, 6, 3, 16)
    torch.testing.assert_close(
        rope(x, start_pos=11),
        rope(x, position_ids=torch.arange(11, 17)),
    )


def test_batched_positions_are_applied_per_sample():
    rope = RotaryEmbedding(16)
    x = torch.randn(2, 6, 3, 16)
    positions = torch.stack((torch.arange(6), torch.arange(10, 16)))
    batched = rope(x, position_ids=positions)
    torch.testing.assert_close(batched[0:1], rope(x[0:1], position_ids=positions[0]))
    torch.testing.assert_close(batched[1:2], rope(x[1:2], position_ids=positions[1]))


@pytest.mark.parametrize(
    "position_ids",
    [torch.arange(5), torch.zeros(2, 6, 1), torch.zeros(3, 6)],
)
def test_invalid_position_shapes_rejected(position_ids):
    with pytest.raises(ValueError):
        RotaryEmbedding(16)(torch.randn(2, 6, 3, 16), position_ids=position_ids)


def test_negative_and_large_positions_are_supported_and_finite():
    rope = RotaryEmbedding(16)
    x = torch.randn(1, 4, 2, 16)
    for positions in (torch.tensor([-3, -2, -1, 0]), torch.arange(10000, 10004)):
        y = rope(x, position_ids=positions)
        assert torch.isfinite(y).all()
        torch.testing.assert_close(
            torch.linalg.vector_norm(y, dim=-1),
            torch.linalg.vector_norm(x, dim=-1),
            atol=1e-5,
            rtol=1e-5,
        )


def test_bfloat16_dtype_and_backward():
    rope = RotaryEmbedding(16, rotary_dim=8)
    x = torch.randn(2, 6, 3, 16, dtype=torch.bfloat16, requires_grad=True)
    y = rope(x)
    y.float().square().mean().backward()
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_float16():
    rope = RotaryEmbedding(16).cuda()
    x = torch.randn(2, 6, 3, 16, device="cuda", dtype=torch.float16)
    y = rope(x)
    assert y.dtype == torch.float16 and y.device.type == "cuda"
