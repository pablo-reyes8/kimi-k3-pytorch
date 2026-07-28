import copy
import io

import pytest
import torch

from src.kda import KDAConfig, KDAState, KimiDeltaAttention
from tests.kda.conftest import tiny_config, tiny_kda


def test_model_state_dict_roundtrip_exact():
    model = tiny_kda().eval()
    clone = KimiDeltaAttention(model.config).eval()
    clone.load_state_dict(model.state_dict())
    x = torch.randn(2, 7, 12)
    first = model(x, mode="chunkwise", output_final_state=True)
    second = clone(x, mode="chunkwise", output_final_state=True)
    torch.testing.assert_close(first.hidden_states, second.hidden_states, rtol=0, atol=0)
    torch.testing.assert_close(
        first.state.recurrent_state, second.state.recurrent_state, rtol=0, atol=0
    )


def test_config_and_state_torch_serialization_roundtrip():
    model = tiny_kda().eval()
    state = model(
        torch.randn(2, 5, 12), mode="recurrent", output_final_state=True
    ).state
    stream = io.BytesIO()
    torch.save({"config": model.config.to_dict(), "state": state}, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    assert KDAConfig.from_dict(restored["config"]) == model.config
    restored_state = restored["state"]
    assert isinstance(restored_state, KDAState)
    torch.testing.assert_close(restored_state.recurrent_state, state.recurrent_state)
    torch.testing.assert_close(
        restored_state.q_conv_state.buffer, state.q_conv_state.buffer
    )


def test_state_clone_has_no_storage_aliases_and_preserves_autograd():
    model = tiny_kda()
    x = torch.randn(1, 3, 12, requires_grad=True)
    state = model(x, mode="recurrent", output_final_state=True).state
    clone = state.clone()
    tensors = (
        (state.recurrent_state, clone.recurrent_state),
        (state.q_conv_state.buffer, clone.q_conv_state.buffer),
        (state.k_conv_state.buffer, clone.k_conv_state.buffer),
        (state.v_conv_state.buffer, clone.v_conv_state.buffer),
        (state.sequence_offset, clone.sequence_offset),
    )
    for original, copied in tensors:
        torch.testing.assert_close(original, copied)
        assert original.data_ptr() != copied.data_ptr()
    clone.recurrent_state.sum().backward()
    assert x.grad is not None


def test_eval_is_deterministic():
    model = tiny_kda().eval()
    x = torch.randn(2, 7, 12)
    assert torch.equal(
        model(x, mode="chunkwise").hidden_states,
        model(x, mode="chunkwise").hidden_states,
    )


def test_module_to_float64_moves_every_parameter_and_output():
    model = tiny_kda().double()
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    output = model(
        torch.randn(1, 4, 12, dtype=torch.float64),
        mode="chunkwise",
        output_final_state=True,
    )
    assert output.hidden_states.dtype == torch.float64
    assert output.state.recurrent_state.dtype == torch.float64


def test_state_rejects_incompatible_recurrent_or_conv_shapes():
    model = tiny_kda()
    state = model(
        torch.randn(2, 3, 12), mode="recurrent", output_final_state=True
    ).state
    wrong = state.clone()
    wrong.recurrent_state = torch.randn(2, 3, 3, 4)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 12), state=wrong, mode="decode")
    wrong = state.clone()
    wrong.q_conv_state.buffer = torch.randn(2, 2, 7)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 12), state=wrong, mode="decode")


def test_qkv_parameters_and_cache_buffers_never_alias():
    model = tiny_kda()
    pointers = {
        model.projections.q_proj.weight.data_ptr(),
        model.projections.k_proj.weight.data_ptr(),
        model.projections.v_proj.weight.data_ptr(),
        model.projections.q_conv.weight.data_ptr(),
        model.projections.k_conv.weight.data_ptr(),
        model.projections.v_conv.weight.data_ptr(),
    }
    assert len(pointers) == 6
    state = model(torch.randn(1, 1, 12), mode="decode").state
    buffer_pointers = {
        state.q_conv_state.buffer.data_ptr(),
        state.k_conv_state.buffer.data_ptr(),
        state.v_conv_state.buffer.data_ptr(),
    }
    assert len(buffer_pointers) == 3
