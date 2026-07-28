import torch

from tests.hybrid_backbone.conftest import tiny_backbone


def explicit_forward(model, hidden_states, mask):
    output = hidden_states
    states = [hidden_states]
    for layer in model.layers:
        output = layer(output, mask).hidden_states
        states.append(output)
    return model.final_norm(output), tuple(states)


def test_backbone_matches_explicit_layer_iteration_exactly():
    model = tiny_backbone(num_hybrid_groups=2).double().eval()
    x = torch.randn(2, 6, 8, dtype=torch.float64)
    mask = torch.ones(2, 6, dtype=torch.bool)
    expected, _ = explicit_forward(model, x, mask)
    actual = model(x).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hidden_state_contract_records_pre_norm_depth_states_and_final_norm():
    model = tiny_backbone().double().eval()
    x = torch.randn(1, 5, 8, dtype=torch.float64)
    mask = torch.ones(1, 5, dtype=torch.bool)
    expected_final, layer_states = explicit_forward(model, x, mask)
    output = model(x, output_hidden_states=True)
    assert len(output.hidden_states) == len(model.layers) + 2
    assert output.hidden_states[0] is x
    for actual, expected in zip(output.hidden_states[:-1], layer_states):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output.hidden_states[-1], expected_final, rtol=0, atol=0
    )


def test_zero_sublayers_leave_pre_final_norm_state_identical_to_input():
    model = tiny_backbone().eval()
    with torch.no_grad():
        for layer in model.layers:
            for parameter in layer.attention.parameters():
                parameter.zero_()
            if layer.ffn is not None:
                for parameter in layer.ffn.parameters():
                    parameter.zero_()
    x = torch.randn(2, 4, 8)
    output = model(x, output_hidden_states=True)
    torch.testing.assert_close(output.hidden_states[-2], x, rtol=0, atol=0)
    torch.testing.assert_close(
        output.last_hidden_state, model.final_norm(x), rtol=0, atol=0
    )


def test_full_and_prefill_outputs_are_exactly_equal():
    model = tiny_backbone(num_hybrid_groups=2).double().eval()
    x = torch.randn(2, 9, 8, dtype=torch.float64)
    full = model(x, mode="full").last_hidden_state
    prefill = model(x, mode="prefill", use_cache=True)
    torch.testing.assert_close(full, prefill.last_hidden_state, rtol=0, atol=0)
    assert prefill.cache.sequence_length == 9


def test_noncontiguous_input_and_edge_t1_b1_are_supported():
    model = tiny_backbone().eval()
    x = torch.randn(1, 1, 16)[..., ::2]
    assert not x.is_contiguous()
    output = model(x, mode="prefill", use_cache=True)
    assert output.last_hidden_state.shape == (1, 1, 8)
    assert torch.isfinite(output.last_hidden_state).all()


def test_invalid_modes_shapes_and_empty_sequences_are_rejected():
    model = tiny_backbone()
    for shape, mode in (
        ((2, 3, 7), "full"),
        ((2, 0, 8), "full"),
        ((2, 3, 8), "unknown"),
    ):
        try:
            model(torch.randn(shape), mode=mode)
        except ValueError:
            pass
        else:
            raise AssertionError(f"shape={shape}, mode={mode} must fail")
