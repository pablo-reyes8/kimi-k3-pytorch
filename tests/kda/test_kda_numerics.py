import pytest
import torch

from src.kda import chunkwise_kda, recurrent_kda
from tests.kda.conftest import random_core, tiny_kda


@pytest.mark.parametrize("mode", ["recurrent", "chunkwise"])
@pytest.mark.parametrize("scale", [0.0, 1e-6, 1.0, 1e3])
def test_full_module_extreme_finite_inputs_remain_finite(mode, scale):
    model = tiny_kda().eval()
    x = torch.randn(2, 17, 12) * scale
    output = model(
        x, mode=mode, output_final_state=True, output_diagnostics=True
    )
    assert torch.isfinite(output.hidden_states).all()
    assert torch.isfinite(output.state.recurrent_state).all()
    assert all(torch.isfinite(value.float()) for value in output.diagnostics.values())


def test_moderately_long_fp32_outputs_states_gradients_are_finite():
    model = tiny_kda(chunk_size=16, secondary_tile_size=8)
    x = torch.randn(1, 128, 12, requires_grad=True)
    output = model(x, mode="chunkwise", output_final_state=True)
    loss = output.hidden_states.square().mean() + output.state.recurrent_state.square().mean()
    loss.backward()
    assert torch.isfinite(output.hidden_states).all()
    assert torch.isfinite(output.state.recurrent_state).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_bfloat16_cpu_recurrent_chunkwise_decode_are_finite_and_close_to_fp32():
    fp32_model = tiny_kda().eval()
    bf16_model = tiny_kda().eval().to(torch.bfloat16)
    bf16_model.load_state_dict(fp32_model.state_dict())
    x = torch.randn(1, 17, 12)
    reference = fp32_model(
        x, mode="recurrent", output_final_state=True
    )
    recurrent = bf16_model(
        x.to(torch.bfloat16), mode="recurrent", output_final_state=True
    )
    chunked = bf16_model(
        x.to(torch.bfloat16), mode="chunkwise", output_final_state=True
    )
    state, pieces = None, []
    for index in range(x.shape[1]):
        decoded = bf16_model(
            x[:, index : index + 1].to(torch.bfloat16),
            state=state,
            mode="decode",
        )
        pieces.append(decoded.hidden_states)
        state = decoded.state
    decode_output = torch.cat(pieces, 1)
    for output in (recurrent.hidden_states, chunked.hidden_states, decode_output):
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output.float()).all()
        torch.testing.assert_close(
            output.float(), reference.hidden_states, rtol=0.08, atol=0.015
        )
    torch.testing.assert_close(
        recurrent.hidden_states, chunked.hidden_states, rtol=0.02, atol=0.002
    )
    torch.testing.assert_close(
        recurrent.hidden_states, decode_output, rtol=0.02, atol=0.002
    )


def test_fp32_chunkwise_error_against_float64_reference_is_small():
    inputs64 = random_core(
        batch=1, tokens=31, heads=2, key_dim=3, value_dim=4
    )
    reference = recurrent_kda(*inputs64, output_final_state=True)
    inputs32 = tuple(value.float() for value in inputs64)
    actual = chunkwise_kda(
        *inputs32,
        output_final_state=True,
        chunk_size=8,
        secondary_tile_size=4,
    )
    torch.testing.assert_close(
        actual.hidden_states.double(), reference.hidden_states,
        rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        actual.final_state.double(), reference.final_state,
        rtol=2e-5, atol=2e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bfloat16_recurrent_chunkwise_decode():
    model = tiny_kda().cuda().to(torch.bfloat16).eval()
    x = torch.randn(1, 17, 12, device="cuda", dtype=torch.bfloat16)
    recurrent = model(x, mode="recurrent", output_final_state=True)
    chunked = model(x, mode="chunkwise", output_final_state=True)
    assert torch.isfinite(recurrent.hidden_states.float()).all()
    assert torch.isfinite(chunked.hidden_states.float()).all()
    torch.testing.assert_close(
        recurrent.hidden_states, chunked.hidden_states, rtol=0.03, atol=0.003
    )

