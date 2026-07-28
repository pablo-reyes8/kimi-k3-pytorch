import pytest
import torch

from src.hybrid_backbone import DenseKimiFFN
from src.kimi_primitives import situ_glu_activation


def test_dense_ffn_matches_exact_situ_glu_equation():
    ffn = DenseKimiFFN(6, 11, dropout=0.0, bias=True).double().eval()
    x = torch.randn(2, 4, 6, dtype=torch.float64)
    transform = ffn.transform
    expected = transform.down_proj(
        situ_glu_activation(
            transform.gate_proj(x),
            transform.up_proj(x),
            transform.beta_gate,
            transform.beta_up,
        )
    )
    torch.testing.assert_close(ffn(x), expected, rtol=0, atol=0)


def test_dense_ffn_interface_is_shape_preserving_and_replaceable():
    ffn = DenseKimiFFN(8, 12)
    x = torch.randn(2, 5, 8)
    assert ffn(x).shape == x.shape
    assert not hasattr(ffn, "experts")
    assert not hasattr(ffn, "router")


def test_dense_ffn_dropout_train_eval_contract():
    ffn = DenseKimiFFN(8, 12, dropout=0.8)
    x = torch.randn(2, 16, 8)
    ffn.train()
    assert not torch.equal(ffn(x), ffn(x))
    ffn.eval()
    torch.testing.assert_close(ffn(x), ffn(x), rtol=0, atol=0)


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_dense_ffn_rejects_invalid_dropout(dropout):
    with pytest.raises(ValueError):
        DenseKimiFFN(8, 12, dropout=dropout)


def test_dense_ffn_all_parameters_and_input_receive_gradient():
    ffn = DenseKimiFFN(8, 12).double()
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    ffn(x).square().sum().backward()
    assert x.grad is not None and torch.count_nonzero(x.grad)
    for name, parameter in ffn.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name
