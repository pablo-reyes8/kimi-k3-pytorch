import pytest
import torch

from data import SyntheticRetrievalConfig, create_synthetic_retrieval_dataloaders
from src.transformer_modules import (
    BaselineTransformerBlock,
    MultiHeadSelfAttention,
    RMSNorm,
    SwiGLUFeedForward,
    TransformerBlockConfig,
)


def make_config(**overrides):
    values = dict(
        d_model=32,
        rms_norm_eps=1e-6,
        n_heads=4,
        head_dim=8,
        attention_dropout=0.0,
        residual_dropout=0.0,
        use_attention_bias=False,
        use_rope=True,
        rope_theta=10_000.0,
        rotary_dim=8,
        max_seq_len=32,
        mlp_hidden_dim=64,
        mlp_expansion_factor=4.0,
        mlp_multiple_of=1,
        mlp_dropout=0.0,
        use_mlp_bias=False,
        init_std=0.02,
    )
    values.update(overrides)
    return TransformerBlockConfig(**values)


def make_block(**overrides):
    return BaselineTransformerBlock(make_config(**overrides))


@pytest.mark.parametrize(
    "override",
    [
        {"d_model": 0},
        {"rms_norm_eps": 0},
        {"max_seq_len": 0},
        {"init_std": 0},
        {"n_heads": 0},
        {"mlp_hidden_dim": 0},
    ],
)
def test_invalid_flat_or_child_configuration_rejected(override):
    with pytest.raises(ValueError):
        make_block(**override)


def test_child_configs_inherit_all_relevant_values():
    config = make_config(
        d_model=48,
        n_heads=6,
        head_dim=8,
        attention_dropout=0.2,
        mlp_hidden_dim=99,
        mlp_dropout=0.3,
    )
    attention = config.to_attention_config()
    mlp = config.to_mlp_config()
    assert (attention.d_model, attention.n_heads, attention.head_dim) == (48, 6, 8)
    assert attention.attention_dropout == 0.2
    assert (mlp.d_model, mlp.hidden_dim, mlp.dropout) == (48, 99, 0.3)


def test_expected_pre_norm_structure_and_initialization():
    block = make_block()
    assert isinstance(block.norm1, RMSNorm)
    assert isinstance(block.attention, MultiHeadSelfAttention)
    assert isinstance(block.norm2, RMSNorm)
    assert isinstance(block.mlp, SwiGLUFeedForward)
    torch.testing.assert_close(block.norm1.weight, torch.ones(32))
    torch.testing.assert_close(block.norm2.weight, torch.ones(32))


@pytest.mark.parametrize("shape", [(2, 8, 32), (0, 8, 32), (2, 0, 32)])
def test_valid_shape_preserved(shape):
    assert make_block()(torch.randn(*shape)).shape == shape


@pytest.mark.parametrize("shape", [(8, 32), (2, 8, 32, 1), (2, 8, 31)])
def test_invalid_input_contract_rejected(shape):
    with pytest.raises(ValueError):
        make_block()(torch.randn(*shape))


def test_sequence_length_limit_enforced():
    with pytest.raises(ValueError, match="max_seq_len"):
        make_block(max_seq_len=4)(torch.randn(1, 5, 32))


def test_need_weights_returns_correct_auxiliary_tensor():
    output, auxiliary = make_block()(torch.randn(2, 8, 32), need_weights=True)
    assert output.shape == (2, 8, 32)
    assert set(auxiliary) == {"attn_weights"}
    assert auxiliary["attn_weights"].shape == (2, 4, 8, 8)


def test_block_is_exact_identity_when_both_sublayers_are_zero():
    block = make_block(use_attention_bias=True, use_mlp_bias=True).eval()
    with torch.no_grad():
        for parameter in block.attention.parameters():
            parameter.zero_()
        for parameter in block.mlp.parameters():
            parameter.zero_()
    x = torch.randn(2, 8, 32)
    torch.testing.assert_close(block(x), x, atol=0, rtol=0)


def test_pre_norm_residual_equation_matches_manual_composition():
    block = make_block().eval()
    x = torch.randn(2, 8, 32)
    after_attention = x + block.attention(block.norm1(x))
    expected = after_attention + block.mlp(block.norm2(after_attention))
    torch.testing.assert_close(block(x), expected)


def test_future_changes_do_not_affect_past_outputs():
    block = make_block().eval()
    first = torch.randn(2, 10, 32)
    second = first.clone()
    second[:, 5:] = torch.randn_like(second[:, 5:])
    torch.testing.assert_close(
        block(first)[:, :5], block(second)[:, :5], atol=1e-6, rtol=1e-5
    )


def test_padding_mask_propagates_to_attention_exactly():
    block = make_block().eval()
    mask = torch.ones(2, 8)
    mask[0, 3] = 0
    mask[1, 5] = 0
    _, auxiliary = block(torch.randn(2, 8, 32), attention_mask=mask, need_weights=True)
    weights = auxiliary["attn_weights"]
    assert torch.count_nonzero(weights[0, :, :, 3]) == 0
    assert torch.count_nonzero(weights[1, :, :, 5]) == 0


def test_rope_position_arguments_propagate():
    block = make_block().eval()
    x = torch.randn(2, 8, 32)
    torch.testing.assert_close(
        block(x, start_pos=20),
        block(x, position_ids=torch.arange(20, 28)),
    )


def test_dropout_train_eval_contract():
    x = torch.randn(4, 12, 32)
    stochastic = make_block(
        attention_dropout=0.5, residual_dropout=0.5, mlp_dropout=0.5
    )
    stochastic.train()
    assert not torch.equal(stochastic(x), stochastic(x))
    stochastic.eval()
    assert torch.equal(stochastic(x), stochastic(x))
    deterministic = make_block().train()
    assert torch.equal(deterministic(x), deterministic(x))


def test_every_parameter_and_input_receives_finite_gradient():
    block = make_block(use_attention_bias=True, use_mlp_bias=True)
    x = torch.randn(2, 7, 32, requires_grad=True)
    block(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_bfloat16_forward_backward():
    block = make_block().to(torch.bfloat16)
    x = torch.randn(2, 7, 32, dtype=torch.bfloat16, requires_grad=True)
    output = block(x)
    output.float().mean().backward()
    assert output.dtype == torch.bfloat16 and torch.isfinite(output.float()).all()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()


def test_synthetic_batch_passes_through_block_with_mask():
    config = SyntheticRetrievalConfig(
        num_train_examples=4,
        num_val_examples=2,
        block_size=24,
        min_filler_tokens=0,
        max_filler_tokens=2,
        num_keys_per_example=1,
        vocab_filler_size=8,
        num_key_types=2,
        num_value_types=4,
        batch_size=2,
    )
    loader, _, tokenizer = create_synthetic_retrieval_dataloaders(config)
    batch = next(iter(loader))
    embedding = torch.nn.Embedding(tokenizer.vocab_size, 32, padding_idx=tokenizer.pad_id)
    output = make_block(max_seq_len=24)(
        embedding(batch["input_ids"]), attention_mask=batch["attention_mask"]
    )
    assert output.shape == (2, 24, 32) and torch.isfinite(output).all()


def test_state_dict_roundtrip_exact():
    first, second = make_block().eval(), make_block().eval()
    second.load_state_dict(first.state_dict())
    x = torch.randn(2, 8, 32)
    torch.testing.assert_close(first(x), second(x), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_backward():
    block = make_block().cuda()
    x = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
    block(x).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
