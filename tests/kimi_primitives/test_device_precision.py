import pytest
import torch

from src.kimi_primitives import (
    CausalShortConv1D,
    FullRankOutputGate,
    HeadwiseRMSNorm,
    SiTUGLU,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_primitives_cuda_forward_backward(dtype):
    situ = SiTUGLU(8, 12).cuda().to(dtype)
    conv = CausalShortConv1D(8, 4).cuda().to(dtype)
    norm = HeadwiseRMSNorm(2, 4).cuda().to(dtype)
    gate = FullRankOutputGate(8).cuda().to(dtype)
    x = torch.randn(2, 5, 8, device="cuda", dtype=dtype, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    convolved = conv(situ(x))
    normalized = norm(convolved.reshape(2, 5, 2, 4)).reshape(2, 5, 8)
    output = gate(normalized, residual)
    output.float().square().mean().backward()
    assert output.dtype == dtype and torch.isfinite(output.float()).all()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()
    assert residual.grad is not None and torch.isfinite(residual.grad.float()).all()

