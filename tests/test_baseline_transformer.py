import pytest
import torch

from src.transformer_modules import (
    BaselineTransformer,
    BaselineTransformerBlock,
    TransformerBlockConfig,
)


def config():
    return TransformerBlockConfig(
        d_model=24,
        n_heads=3,
        head_dim=8,
        mlp_hidden_dim=48,
        max_seq_len=16,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
    )


@pytest.mark.parametrize("n_layers", [0, -1])
def test_invalid_layer_count_rejected(n_layers):
    with pytest.raises(ValueError):
        BaselineTransformer(config(), n_layers)


def test_stack_contains_independent_blocks():
    stack = BaselineTransformer(config(), 3)
    assert len(stack.layers) == 3
    assert all(isinstance(layer, BaselineTransformerBlock) for layer in stack.layers)
    assert len({id(layer) for layer in stack.layers}) == 3
    first_parameter_ids = {id(p) for p in stack.layers[0].parameters()}
    second_parameter_ids = {id(p) for p in stack.layers[1].parameters()}
    assert first_parameter_ids.isdisjoint(second_parameter_ids)


def test_forward_matches_explicit_layer_iteration():
    stack = BaselineTransformer(config(), 3).eval()
    x = torch.randn(2, 12, 24)
    expected = x
    for layer in stack.layers:
        expected = layer(expected)
    torch.testing.assert_close(stack(x), expected)


def test_mask_and_positions_propagate_to_every_layer():
    stack = BaselineTransformer(config(), 2).eval()
    x = torch.randn(2, 12, 24)
    mask = torch.ones(2, 12)
    mask[:, 4] = 0
    positions = torch.arange(10, 22)
    expected = x
    for layer in stack.layers:
        expected = layer(expected, attention_mask=mask, position_ids=positions)
    torch.testing.assert_close(
        stack(x, attention_mask=mask, position_ids=positions), expected
    )


def test_stack_is_causal_end_to_end():
    stack = BaselineTransformer(config(), 3).eval()
    first = torch.randn(2, 12, 24)
    second = first.clone()
    second[:, 6:] = torch.randn_like(second[:, 6:])
    torch.testing.assert_close(
        stack(first)[:, :6], stack(second)[:, :6], atol=1e-5, rtol=1e-5
    )


def test_gradient_reaches_every_layer():
    stack = BaselineTransformer(config(), 3)
    x = torch.randn(2, 12, 24, requires_grad=True)
    stack(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in stack.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_zero_sublayers_make_entire_stack_identity():
    stack = BaselineTransformer(config(), 3).eval()
    with torch.no_grad():
        for layer in stack.layers:
            for parameter in layer.attention.parameters():
                parameter.zero_()
            for parameter in layer.mlp.parameters():
                parameter.zero_()
    x = torch.randn(2, 12, 24)
    torch.testing.assert_close(stack(x), x, atol=0, rtol=0)


def test_state_dict_roundtrip_exact():
    first = BaselineTransformer(config(), 2).eval()
    second = BaselineTransformer(config(), 2).eval()
    second.load_state_dict(first.state_dict())
    x = torch.randn(2, 12, 24)
    torch.testing.assert_close(first(x), second(x), atol=0, rtol=0)
