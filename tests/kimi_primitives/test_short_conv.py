import copy

import pytest
import torch

from src.kimi_primitives import CausalShortConv1D, ShortConvState


def manual_causal_conv(x, weight, bias=None):
    batch, tokens, channels = x.shape
    kernel = weight.shape[1]
    output = torch.zeros_like(x)
    for b in range(batch):
        for t in range(tokens):
            for c in range(channels):
                value = x.new_zeros(())
                for lag in range(kernel):
                    if t - lag >= 0:
                        value = value + weight[c, lag] * x[b, t - lag, c]
                if bias is not None:
                    value = value + bias[c]
                output[b, t, c] = value
    return output


def controlled_conv(channels=2, kernel_size=3, bias=True):
    module = CausalShortConv1D(channels, kernel_size, bias=bias)
    with torch.no_grad():
        module.weight.copy_(
            torch.arange(1, channels * kernel_size + 1).reshape(
                channels, kernel_size
            )
        )
        if module.bias is not None:
            module.bias.copy_(torch.linspace(-0.5, 0.5, channels))
    return module


def test_short_conv_matches_explicit_loop_and_lag_orientation():
    module = controlled_conv()
    x = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    expected = manual_causal_conv(x, module.weight, module.bias)
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("changed_after", range(6))
def test_short_conv_is_causal_for_every_prefix(changed_after):
    module = controlled_conv(channels=3, kernel_size=4).eval()
    first = torch.randn(2, 7, 3)
    second = first.clone()
    second[:, changed_after + 1 :] = torch.randn_like(second[:, changed_after + 1 :]) * 100
    torch.testing.assert_close(
        module(first)[:, : changed_after + 1],
        module(second)[:, : changed_after + 1],
        rtol=0,
        atol=0,
    )


def test_impulse_response_identifies_exact_lags_without_kernel_reversal():
    module = CausalShortConv1D(1, 4, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1.0, 2.0, 4.0, 8.0]]))
    x = torch.zeros(1, 8, 1)
    x[:, 2] = 1
    output = module(x).flatten()
    expected = torch.tensor([0, 0, 1, 2, 4, 8, 0, 0], dtype=torch.float32)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("kernel_size", [1, 2, 4, 7])
def test_token_decode_matches_full_sequence(batch, kernel_size):
    module = CausalShortConv1D(3, kernel_size, bias=True)
    x = torch.randn(batch, 11, 3)
    expected = module(x)
    state = None
    pieces = []
    for index in range(x.shape[1]):
        output, state = module(
            x[:, index : index + 1], state, return_state=True
        )
        pieces.append(output)
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), expected, rtol=1e-5, atol=1e-6
    )
    assert state.buffer.shape == (batch, kernel_size - 1, 3)


@pytest.mark.parametrize(
    "chunks", [[13], [2, 11], [1] * 13, [3, 1, 5, 4], [0, 4, 0, 9]]
)
def test_irregular_chunking_invariance(chunks):
    module = CausalShortConv1D(4, 5, bias=True)
    x = torch.randn(2, 13, 4)
    expected = module(x)
    state, offset, pieces = None, 0, []
    for size in chunks:
        output, state = module(
            x[:, offset : offset + size], state, return_state=True
        )
        pieces.append(output)
        offset += size
    assert offset == x.shape[1]
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), expected, rtol=1e-5, atol=1e-6
    )


def test_new_none_state_resets_history():
    module = CausalShortConv1D(2, 4)
    prefix, suffix = torch.randn(1, 5, 2), torch.randn(1, 3, 2)
    _, state = module(prefix, return_state=True)
    continued = module(suffix, state)
    reset = module(suffix, None)
    assert not torch.allclose(continued, reset)
    torch.testing.assert_close(reset, CausalShortConv1D.forward(module, suffix))


def test_input_state_is_not_mutated_or_aliased():
    module = CausalShortConv1D(2, 4)
    initial = ShortConvState(torch.randn(2, 3, 2))
    saved = initial.buffer.clone()
    _, next_state = module(torch.randn(2, 5, 2), initial, return_state=True)
    torch.testing.assert_close(initial.buffer, saved, rtol=0, atol=0)
    assert next_state.buffer.data_ptr() != initial.buffer.data_ptr()


def test_channel_independence_is_exact():
    module = CausalShortConv1D(3, 3, bias=False)
    x = torch.randn(1, 6, 3)
    changed = x.clone()
    changed[..., 1] += 100
    first, second = module(x), module(changed)
    torch.testing.assert_close(first[..., 0], second[..., 0], rtol=0, atol=0)
    torch.testing.assert_close(first[..., 2], second[..., 2], rtol=0, atol=0)
    assert not torch.equal(first[..., 1], second[..., 1])


def test_batch_independence_is_exact():
    module = CausalShortConv1D(2, 3)
    x = torch.randn(3, 7, 2)
    changed = x.clone()
    changed[1] += 100
    first, second = module(x), module(changed)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def test_kernel_size_one_is_pointwise_per_channel():
    module = controlled_conv(channels=2, kernel_size=1)
    x = torch.randn(2, 5, 2)
    expected = x * module.weight[:, 0] + module.bias
    output, state = module(x, return_state=True)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert state.buffer.shape == (2, 0, 2)


def test_empty_chunk_preserves_state_value_without_aliasing():
    module = CausalShortConv1D(2, 3)
    state = ShortConvState(torch.randn(1, 2, 2))
    output, next_state = module(
        torch.empty(1, 0, 2), state, return_state=True
    )
    assert output.shape == (1, 0, 2)
    torch.testing.assert_close(next_state.buffer, state.buffer, rtol=0, atol=0)
    assert next_state.buffer.data_ptr() != state.buffer.data_ptr()


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: CausalShortConv1D(0, 3),
        lambda: CausalShortConv1D(2, 0),
    ],
)
def test_invalid_configuration_rejected(constructor):
    with pytest.raises(ValueError):
        constructor()


@pytest.mark.parametrize(
    "buffer",
    [
        torch.randn(1, 2),
        torch.randn(1, 3, 2),
        torch.randn(2, 2, 3),
    ],
)
def test_incompatible_state_shape_rejected(buffer):
    module = CausalShortConv1D(2, 3)
    with pytest.raises(ValueError):
        module(torch.randn(1, 4, 2), ShortConvState(buffer))


def test_state_dtype_mismatch_rejected():
    module = CausalShortConv1D(2, 3).double()
    state = ShortConvState(torch.zeros(1, 2, 2, dtype=torch.float32))
    with pytest.raises(ValueError, match="dtype"):
        module(torch.randn(1, 4, 2, dtype=torch.float64), state)


def test_short_conv_gradcheck_input_weight_and_bias():
    module = CausalShortConv1D(2, 3, bias=True).double()
    x = torch.randn(1, 4, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), fast_mode=True)


def test_cached_chunks_preserve_gradients_through_state():
    module = CausalShortConv1D(2, 3)
    first = torch.randn(1, 2, 2, requires_grad=True)
    second = torch.randn(1, 2, 2, requires_grad=True)
    _, state = module(first, return_state=True)
    output = module(second, state)
    output.square().mean().backward()
    assert first.grad is not None and first.grad.abs().sum() > 0
    assert second.grad is not None and second.grad.abs().sum() > 0
    assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0


def test_state_dict_roundtrip_is_exact():
    module = CausalShortConv1D(3, 4, bias=True).eval()
    clone = copy.deepcopy(module).eval()
    clone.load_state_dict(module.state_dict())
    x = torch.randn(2, 6, 3)
    torch.testing.assert_close(module(x), clone(x), rtol=0, atol=0)


def test_bfloat16_full_and_cached_paths_are_finite_and_close():
    module = CausalShortConv1D(3, 4, bias=True).to(torch.bfloat16)
    x = torch.randn(2, 7, 3, dtype=torch.bfloat16)
    full = module(x)
    first, state = module(x[:, :3], return_state=True)
    second = module(x[:, 3:], state)
    cached = torch.cat((first, second), dim=1)
    assert full.dtype == torch.bfloat16 and torch.isfinite(full.float()).all()
    torch.testing.assert_close(cached, full, rtol=0.02, atol=0.02)

