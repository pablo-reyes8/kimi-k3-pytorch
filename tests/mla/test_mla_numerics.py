import pytest
import torch

from src.mla import mla_attention
from tests.mla.conftest import random_qkv, tiny_mla


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_fp32_forward_backward_attention_and_cache_are_finite(backend):
    model = tiny_mla(attention_backend=backend)
    x = torch.randn(2, 32, 12, requires_grad=True)
    output = model(x, use_cache=True, output_attentions=True)
    output.hidden_states.square().mean().backward()
    assert torch.isfinite(output.hidden_states).all()
    assert torch.isfinite(output.attentions).all()
    assert torch.isfinite(output.cache.latent_kv).all()
    assert torch.isfinite(x.grad).all()


def test_keep_output_fp32_policy_is_explicit_for_bfloat16_core():
    q, k, v = random_qkv(dtype=torch.bfloat16)
    kept = mla_attention(q, k, v, backend="manual", keep_output_fp32=True)
    restored = mla_attention(q, k, v, backend="manual", keep_output_fp32=False)
    assert kept.dtype == torch.float32
    assert restored.dtype == torch.bfloat16
    reference = mla_attention(
        q.float(), k.float(), v.float(), backend="manual", keep_output_fp32=True
    )
    torch.testing.assert_close(kept, reference, rtol=0, atol=0)


def test_full_bfloat16_returns_model_dtype_and_matches_fp32_reasonably():
    fp32 = tiny_mla(attention_backend="manual").eval()
    bf16 = tiny_mla(attention_backend="manual").eval()
    bf16.load_state_dict(fp32.state_dict())
    bf16 = bf16.to(torch.bfloat16)
    x = torch.randn(2, 8, 12)
    reference = fp32(x).hidden_states
    actual = bf16(x.to(torch.bfloat16)).hidden_states
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference, rtol=0.04, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("keep_fp32", [True, False])
def test_cuda_bfloat16_full_prefill_decode(keep_fp32):
    model = tiny_mla(
        keep_attention_output_fp32=keep_fp32
    ).cuda().bfloat16().eval()
    x = torch.randn(2, 6, 12, device="cuda", dtype=torch.bfloat16)
    full = model(x).hidden_states
    prefill = model(x[:, :5], use_cache=True)
    decode = model(x[:, 5:], cache=prefill.cache, use_cache=True)
    torch.testing.assert_close(decode.hidden_states, full[:, 5:], rtol=0.03, atol=0.003)
