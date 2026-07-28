import torch

from src.mla import GatedMLA
from tests.mla.conftest import tiny_config, tiny_mla


def test_module_tree_contains_no_positional_or_rotary_module():
    model = tiny_mla()
    forbidden = ("rope", "rotary", "position", "relative")
    names = [name.lower() for name, _ in model.named_modules()]
    assert not any(token in name for name in names for token in forbidden)


def test_forward_requires_no_position_ids_and_explicitly_rejects_them():
    model = tiny_mla()
    x = torch.randn(1, 3, 12)
    assert model(x).hidden_states.shape == x.shape
    try:
        model(x, position_ids=torch.arange(3)[None])
    except ValueError as error:
        assert "NoPE" in str(error)
    else:
        raise AssertionError("position_ids must not be silently accepted")


def test_noncausal_core_is_token_permutation_equivariant():
    model = GatedMLA(tiny_config(attention_backend="manual")).double().eval()
    x = torch.randn(2, 6, 12, dtype=torch.float64)
    projected = model.projections(x)
    from src.mla import mla_attention
    baseline = mla_attention(
        projected.query, projected.key, projected.value, is_causal=False,
        backend="manual",
    )
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    permuted = model.projections(x[:, permutation])
    actual = mla_attention(
        permuted.query, permuted.key, permuted.value, is_causal=False,
        backend="manual",
    )
    torch.testing.assert_close(actual, baseline[:, permutation], rtol=1e-12, atol=1e-12)


def test_no_hidden_residual_connection():
    model = tiny_mla()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    x = torch.randn(2, 4, 12)
    assert torch.count_nonzero(model(x).hidden_states) == 0
