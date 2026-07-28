from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from data import SyntheticRetrievalConfig, create_synthetic_retrieval_dataloaders
from src import BaselineCausalLM, BaselineCausalLMConfig, CausalLMOutput
from src.transformer_modules import BaselineTransformer, RMSNorm, TokenEmbedding


def make_config(**overrides):
    values = dict(
        vocab_size=48,
        d_model=32,
        n_layers=2,
        n_heads=4,
        mlp_hidden_dim=64,
        max_seq_len=32,
        pad_token_id=0,
        tie_embeddings=True,
        use_rope=True,
        rope_theta=10_000.0,
        dropout=0.0,
        rms_norm_eps=1e-6,
        init_std=0.02,
    )
    values.update(overrides)
    return BaselineCausalLMConfig(**values)


def make_model(**overrides):
    torch.manual_seed(0)
    return BaselineCausalLM(make_config(**overrides))


@pytest.mark.parametrize(
    "override",
    [
        {"vocab_size": 0},
        {"d_model": 0},
        {"n_layers": 0},
        {"n_heads": 0},
        {"d_model": 30, "n_heads": 4},
        {"max_seq_len": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"pad_token_id": -1},
        {"pad_token_id": 48},
    ],
)
def test_invalid_configuration_rejected(override):
    with pytest.raises(ValueError):
        BaselineCausalLM(make_config(**override))


def test_child_config_translation_is_explicit_and_consistent():
    config = make_config(dropout=0.2, use_rope=False, mlp_hidden_dim=77)
    embedding = config.embedding_config()
    block = config.block_config()
    assert (embedding.vocab_size, embedding.d_model) == (48, 32)
    assert embedding.embedding_dropout == 0.2
    assert (block.d_model, block.mlp_hidden_dim) == (32, 77)
    assert block.attention_dropout == block.residual_dropout == block.mlp_dropout == 0.2
    assert block.use_rope is False


def test_expected_model_structure_and_layer_count():
    model = make_model(n_layers=3)
    assert isinstance(model.embedding, TokenEmbedding)
    assert isinstance(model.backbone, BaselineTransformer)
    assert isinstance(model.final_norm, RMSNorm)
    assert isinstance(model.lm_head, torch.nn.Linear)
    assert len(model.backbone.layers) == 3


@pytest.mark.parametrize("tied", [True, False])
def test_embedding_tying_policy(tied):
    model = make_model(tie_embeddings=tied)
    assert (model.lm_head.weight is model.embedding.weight) is tied


def test_untied_head_initialization_is_finite_and_has_requested_std():
    model = make_model(tie_embeddings=False, init_std=0.03)
    weight = model.lm_head.weight.detach()
    assert torch.isfinite(weight).all()
    assert abs(weight.std(unbiased=False).item() - 0.03) < 0.005


def test_forward_returns_strict_output_contract():
    model = make_model().eval()
    ids = torch.randint(1, 48, (2, 12))
    output = model(ids)
    assert isinstance(output, CausalLMOutput)
    assert output.logits.shape == (2, 12, 48)
    assert output.loss is None
    assert output.auxiliary_losses == {}
    assert output.metrics == {}


def test_loss_matches_manual_cross_entropy_with_shifted_label_contract():
    model = make_model().eval()
    ids = torch.randint(1, 48, (2, 12))
    labels = torch.randint(1, 48, (2, 12))
    labels[0, -2:] = 0
    output = model(ids, labels=labels)
    expected = F.cross_entropy(
        output.logits.reshape(-1, 48), labels.reshape(-1), ignore_index=0
    )
    assert output.loss is not None and output.loss.ndim == 0
    torch.testing.assert_close(output.loss, expected)


def test_label_changes_at_same_position_change_loss_without_internal_shift():
    model = make_model(pad_token_id=None).eval()
    ids = torch.randint(0, 48, (1, 8))
    labels = torch.randint(0, 48, (1, 8))
    output = model(ids, labels=labels)
    manual = F.cross_entropy(output.logits[0, 3:4], labels[0, 3:4])
    labels_ignore = torch.full_like(labels, -100)
    labels_ignore[0, 3] = labels[0, 3]
    focused = model(ids, labels=labels_ignore)
    torch.testing.assert_close(focused.loss, manual)


@pytest.mark.parametrize(
    "bad_ids", [torch.ones(8, dtype=torch.long), torch.ones(2, 8, 1, dtype=torch.long)]
)
def test_invalid_input_rank_rejected(bad_ids):
    with pytest.raises(ValueError):
        make_model()(bad_ids)


def test_embedding_layer_rejects_invalid_ids_and_sequence_length():
    model = make_model(max_seq_len=8)
    with pytest.raises(ValueError):
        model(torch.ones(1, 9, dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.tensor([[48]]))


def test_labels_mask_and_embeddings_validate_shapes():
    model = make_model()
    ids = torch.ones(2, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="labels"):
        model(ids, labels=torch.ones(2, 7, dtype=torch.long))
    with pytest.raises(ValueError, match="attention_mask"):
        model(ids, attention_mask=torch.ones(2, 7))
    with pytest.raises(ValueError, match="hidden_states"):
        model.forward_embeddings(torch.randn(2, 8, 31))
    with pytest.raises(ValueError, match="sequence"):
        model.forward_embeddings(torch.randn(2, 33, 32))


def test_auto_padding_mask_equals_explicit_mask():
    model = make_model().eval()
    ids = torch.randint(1, 48, (2, 12))
    ids[0, -3:] = 0
    automatic = model(ids).logits
    explicit = model(ids, attention_mask=ids.ne(0)).logits
    torch.testing.assert_close(automatic, explicit)


def test_forward_embeddings_exactly_matches_token_forward():
    model = make_model().eval()
    ids = torch.randint(1, 48, (2, 12))
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        token_output = model(ids, attention_mask=mask)
        embedding_output = model.forward_embeddings(
            model.embedding(ids), attention_mask=mask
        )
    torch.testing.assert_close(token_output.logits, embedding_output.logits)


def test_forward_embeddings_accepts_continuous_non_token_inputs_and_gradients():
    model = make_model()
    embeddings = torch.randn(2, 10, 32, requires_grad=True)
    labels = torch.randint(1, 48, (2, 10))
    output = model.forward_embeddings(embeddings, labels=labels)
    output.loss.backward()
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()


def test_end_to_end_causality():
    model = make_model().eval()
    first = torch.randint(1, 48, (2, 16))
    second = first.clone()
    second[:, 8:] = torch.randint(1, 48, (2, 8))
    torch.testing.assert_close(
        model(first).logits[:, :8],
        model(second).logits[:, :8],
        atol=1e-5,
        rtol=1e-5,
    )


def test_every_trainable_parameter_receives_finite_gradient_tied_and_untied():
    for tied in (True, False):
        model = make_model(tie_embeddings=tied)
        ids = torch.randint(1, 48, (2, 12))
        labels = torch.randint(1, 48, (2, 12))
        model(ids, labels=labels).loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{tied=} {name}"
            assert torch.isfinite(parameter.grad).all(), f"{tied=} {name}"


def test_num_parameters_matches_unique_pytorch_parameters():
    model = make_model(tie_embeddings=True)
    expected = sum(parameter.numel() for parameter in model.parameters())
    assert model.num_parameters() == expected
    frozen = next(model.backbone.parameters())
    frozen.requires_grad_(False)
    expected_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert model.num_parameters(trainable_only=True) == expected_trainable


def test_bfloat16_forward_and_loss_are_finite():
    model = make_model().to(torch.bfloat16)
    ids = torch.randint(1, 48, (2, 12))
    labels = torch.randint(1, 48, (2, 12))
    output = model(ids, labels=labels)
    assert output.logits.dtype == torch.bfloat16
    assert torch.isfinite(output.logits.float()).all()
    assert torch.isfinite(output.loss)


def test_synthetic_retrieval_forward_backward_is_download_free():
    data_config = SyntheticRetrievalConfig(
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
        seed=7,
    )
    loader, _, tokenizer = create_synthetic_retrieval_dataloaders(data_config)
    batch = next(iter(loader))
    model = BaselineCausalLM(
        BaselineCausalLMConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=24,
            n_layers=2,
            n_heads=3,
            mlp_hidden_dim=48,
            max_seq_len=24,
            pad_token_id=tokenizer.pad_id,
        )
    )
    output = model(**batch)
    assert output.logits.shape == (2, 24, tokenizer.vocab_size)
    output.loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_model_state_and_dataclass_config_roundtrip_exact():
    first = make_model().eval()
    restored_config = BaselineCausalLMConfig(**asdict(first.config))
    second = BaselineCausalLM(restored_config).eval()
    second.load_state_dict(first.state_dict())
    ids = torch.randint(1, 48, (2, 12))
    torch.testing.assert_close(first(ids).logits, second(ids).logits, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_backward():
    model = make_model().cuda()
    ids = torch.randint(1, 48, (2, 12), device="cuda")
    labels = torch.randint(1, 48, (2, 12), device="cuda")
    output = model(ids, labels=labels)
    output.loss.backward()
    assert output.logits.device.type == "cuda"
