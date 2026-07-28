import io

import pytest
import torch

from src import (
    AttentionModuleOutput,
    KDAProjectionOutput,
    PrimitiveAttentionPostprocess,
    ShortConvState,
    combine_heads,
    split_heads,
)


@pytest.mark.parametrize(
    "shape,heads", [((2, 3, 12), 3), ((1, 7, 16), 8), ((0, 3, 8), 2)]
)
def test_split_combine_roundtrip_is_exact(shape, heads):
    hidden = torch.randn(shape)
    split = split_heads(hidden, heads)
    assert split.shape == (shape[0], shape[1], heads, shape[2] // heads)
    restored = combine_heads(split)
    torch.testing.assert_close(restored, hidden, rtol=0, atol=0)
    assert restored.data_ptr() == hidden.data_ptr()


def test_combine_split_roundtrip_is_exact():
    heads = torch.randn(2, 3, 4, 5)
    combined = combine_heads(heads)
    restored = split_heads(combined, 4)
    torch.testing.assert_close(restored, heads, rtol=0, atol=0)


def test_head_channel_order_has_no_hidden_permutation():
    identifiable = torch.arange(24).reshape(1, 2, 3, 4)
    combined = combine_heads(identifiable)
    expected = torch.tensor(
        [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
          [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]]
    )
    torch.testing.assert_close(combined, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "function,args",
    [
        (split_heads, (torch.randn(2, 8), 2)),
        (split_heads, (torch.randn(2, 3, 7), 2)),
        (split_heads, (torch.randn(2, 3, 8), 0)),
        (combine_heads, (torch.randn(2, 3, 8),)),
        (combine_heads, (torch.randn(2, 3, 0, 4),)),
    ],
)
def test_head_utils_reject_ambiguous_shapes(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_head_utils_preserve_autograd():
    hidden = torch.randn(2, 3, 12, requires_grad=True)
    combine_heads(split_heads(hidden, 3)).square().mean().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_short_conv_state_construction_clone_and_no_aliasing():
    buffer = torch.randn(2, 3, 4, requires_grad=True)
    state = ShortConvState(buffer)
    cloned = state.clone()
    torch.testing.assert_close(cloned.buffer, buffer, rtol=0, atol=0)
    assert cloned.buffer.data_ptr() != buffer.data_ptr()
    cloned.buffer.sum().backward()
    assert buffer.grad is not None


@pytest.mark.parametrize("value", [None, torch.randn(2, 3), [1, 2]])
def test_short_conv_state_rejects_invalid_buffer(value):
    with pytest.raises((TypeError, ValueError)):
        ShortConvState(value)


def test_kda_projection_output_validates_shared_contract_without_detaching():
    q = torch.randn(2, 3, 4, requires_grad=True)
    output = KDAProjectionOutput(
        q=q,
        k=torch.randn_like(q),
        v=torch.randn_like(q),
        beta=torch.randn(2, 3, 1),
        alpha=torch.randn(2, 3, 1),
    )
    assert output.q is q
    output.q.sum().backward()
    assert q.grad is not None


def test_kda_projection_output_rejects_incompatible_shapes():
    base = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="identical"):
        KDAProjectionOutput(
            base, torch.randn(2, 3, 5), base, torch.randn(2, 3, 1),
            torch.randn(2, 3, 1)
        )
    with pytest.raises(ValueError, match="batch and token"):
        KDAProjectionOutput(
            base, base, base, torch.randn(2, 4, 1), torch.randn(2, 3, 1)
        )


def test_attention_output_defaults_are_independent_and_immutable_none():
    first = AttentionModuleOutput(torch.randn(2, 3, 4))
    second = AttentionModuleOutput(torch.randn(2, 3, 4))
    assert first.state is None and second.state is None
    assert first.diagnostics is None and second.diagnostics is None


def test_attention_output_preserves_tensors_and_validates_diagnostics():
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    diagnostic = torch.randn(2, 3)
    output = AttentionModuleOutput(hidden, diagnostics={"beta": diagnostic})
    assert output.hidden_states is hidden
    assert output.diagnostics["beta"] is diagnostic
    output.hidden_states.sum().backward()
    assert hidden.grad is not None
    with pytest.raises(TypeError):
        AttentionModuleOutput(torch.randn(2, 3, 4), diagnostics={"bad": 1})


@pytest.mark.parametrize("hidden", [torch.randn(2, 4), "not a tensor"])
def test_attention_output_rejects_invalid_hidden_states(hidden):
    with pytest.raises((TypeError, ValueError)):
        AttentionModuleOutput(hidden)


def test_dataclasses_roundtrip_through_torch_serialization():
    state = ShortConvState(torch.randn(2, 3, 4))
    output = AttentionModuleOutput(
        torch.randn(2, 5, 8),
        state=state,
        diagnostics={"gate": torch.randn(2, 5, 8)},
    )
    stream = io.BytesIO()
    torch.save(output, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    torch.testing.assert_close(
        restored.hidden_states, output.hidden_states, rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored.state.buffer, state.buffer, rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored.diagnostics["gate"], output.diagnostics["gate"], rtol=0, atol=0
    )


def test_postprocess_matches_manual_end_to_end_composition():
    module = PrimitiveAttentionPostprocess(
        3, 4, gate_bias=True, output_bias=True
    )
    heads = torch.randn(2, 5, 3, 4)
    residual = torch.randn(2, 5, 12)
    normalized = module.norm(heads)
    combined = normalized.reshape(2, 5, 12)
    expected_gate = torch.sigmoid(module.output_gate.gate_proj(residual))
    expected = module.output_gate.output_proj(expected_gate * combined)
    actual, gate = module(heads, residual, return_gate=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(gate, expected_gate, rtol=0, atol=0)


def test_postprocess_preserves_identifiable_head_order():
    module = PrimitiveAttentionPostprocess(
        2, 3, norm_affine=False, gate_bias=True
    )
    with torch.no_grad():
        module.output_gate.gate_proj.weight.zero_()
        module.output_gate.gate_proj.bias.fill_(20)
        module.output_gate.output_proj.weight.copy_(torch.eye(6))
    heads = torch.tensor(
        [[[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]]]
    )
    expected = combine_heads(module.norm(heads))
    actual = module(heads, torch.zeros(1, 1, 6))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual[0, 0, 0] < actual[0, 0, 3]


def test_postprocess_end_to_end_gradients_reach_everything():
    module = PrimitiveAttentionPostprocess(
        3, 4, gate_bias=True, output_bias=True
    )
    heads = torch.randn(2, 5, 3, 4, requires_grad=True)
    residual = torch.randn(2, 5, 12, requires_grad=True)
    module(heads, residual).square().mean().backward()
    assert heads.grad is not None and heads.grad.abs().sum() > 0
    assert residual.grad is not None and residual.grad.abs().sum() > 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name
        assert torch.isfinite(parameter.grad).all(), name


def test_postprocess_bfloat16_composition():
    module = PrimitiveAttentionPostprocess(2, 4).to(torch.bfloat16)
    heads = torch.randn(2, 3, 2, 4, dtype=torch.bfloat16)
    residual = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    output = module(heads, residual)
    assert output.shape == residual.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()


def test_root_public_api_exposes_stable_primitives():
    import src

    for name in (
        "SiTUGLU",
        "CausalShortConv1D",
        "HeadwiseRMSNorm",
        "FullRankOutputGate",
        "PrimitiveAttentionPostprocess",
        "ShortConvState",
        "KDAProjectionOutput",
        "AttentionModuleOutput",
        "split_heads",
        "combine_heads",
    ):
        assert hasattr(src, name), name


def test_no_deepseek_runtime_imports_in_primitive_source():
    from pathlib import Path

    source_root = Path(__file__).parents[2] / "src" / "kimi_primitives"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in source_root.glob("*.py")
    ).lower()
    assert "deepseek" not in source

