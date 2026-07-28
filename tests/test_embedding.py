import math

import pytest
import torch

from data import SyntheticRetrievalConfig, create_synthetic_retrieval_dataloaders
from src.transformer_modules import EmbeddingConfig, TokenEmbedding


@pytest.fixture
def config():
    return EmbeddingConfig(
        vocab_size=128,
        d_model=32,
        pad_token_id=0,
        max_seq_len=64,
        embedding_dropout=0.0,
        scale_embeddings=False,
        init_std=0.02,
        tie_word_embeddings=True,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("vocab_size", 0),
        ("vocab_size", -1),
        ("d_model", 0),
        ("d_model", -1),
        ("max_seq_len", 0),
        ("max_seq_len", -1),
        ("embedding_dropout", -0.1),
        ("embedding_dropout", 1.0),
        ("init_std", 0.0),
        ("pad_token_id", -1),
        ("pad_token_id", 128),
    ],
)
def test_invalid_configuration_rejected(config, field, value):
    values = vars(config).copy()
    values[field] = value
    with pytest.raises(ValueError):
        TokenEmbedding(EmbeddingConfig(**values))


def test_structure_weight_property_and_initialization(config):
    torch.manual_seed(123)
    embedding = TokenEmbedding(config)
    assert embedding.weight is embedding.token_embedding.weight
    assert embedding.weight.shape == (config.vocab_size, config.d_model)
    assert torch.count_nonzero(embedding.weight[config.pad_token_id]) == 0
    non_pad = embedding.weight.detach()[1:]
    assert torch.isfinite(non_pad).all()
    assert abs(non_pad.mean().item()) < config.init_std
    assert abs(non_pad.std(unbiased=False).item() - config.init_std) < config.init_std / 3


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_integer_inputs_return_expected_shape_and_float_values(config, dtype):
    embedding = TokenEmbedding(config)
    ids = torch.randint(1, config.vocab_size, (4, 16), dtype=dtype)
    output = embedding(ids)
    assert output.shape == (4, 16, config.d_model)
    assert output.dtype.is_floating_point
    assert torch.isfinite(output).all()


def test_dict_batch_matches_tensor_input(config):
    embedding = TokenEmbedding(config).eval()
    ids = torch.randint(1, config.vocab_size, (2, 8))
    torch.testing.assert_close(embedding(ids), embedding({"input_ids": ids}))
    with pytest.raises(KeyError):
        embedding({"labels": ids})


@pytest.mark.parametrize(
    "bad_input,error",
    [
        (torch.ones(8), ValueError),
        (torch.ones(2, 8, 1), ValueError),
        (torch.ones(2, 8, dtype=torch.float32), TypeError),
        ([[1, 2]], TypeError),
    ],
)
def test_invalid_input_type_rank_or_dtype_rejected(config, bad_input, error):
    with pytest.raises(error):
        TokenEmbedding(config)(bad_input)


def test_sequence_and_token_range_validation(config):
    embedding = TokenEmbedding(config)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        embedding(torch.ones(1, 65, dtype=torch.long))
    with pytest.raises(ValueError, match="negative"):
        embedding(torch.tensor([[1, -1]]))
    with pytest.raises(ValueError, match="vocab_size"):
        embedding(torch.tensor([[1, 128]]))


@pytest.mark.parametrize("scale", [False, True])
def test_forward_matches_raw_lookup_with_configured_scaling(config, scale):
    values = vars(config).copy()
    values["scale_embeddings"] = scale
    embedding = TokenEmbedding(EmbeddingConfig(**values)).eval()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    expected = embedding.token_embedding(ids)
    if scale:
        expected = expected * math.sqrt(config.d_model)
    torch.testing.assert_close(embedding(ids), expected)


@pytest.mark.parametrize("scale", [False, True])
def test_padding_outputs_remain_exactly_zero(config, scale):
    values = vars(config).copy()
    values["scale_embeddings"] = scale
    embedding = TokenEmbedding(EmbeddingConfig(**values))
    ids = torch.tensor([[0, 1, 2], [3, 0, 4]])
    output = embedding(ids)
    assert torch.count_nonzero(output[ids == 0]) == 0


def test_padding_gradient_zero_and_used_token_gradient_nonzero(config):
    embedding = TokenEmbedding(config)
    ids = torch.tensor([[0, 7, 2], [7, 0, 4]])
    embedding(ids).sum().backward()
    assert torch.count_nonzero(embedding.weight.grad[0]) == 0
    assert embedding.weight.grad[7].abs().sum() > 0


def test_dropout_train_eval_contract(config):
    values = vars(config).copy()
    values["embedding_dropout"] = 0.5
    embedding = TokenEmbedding(EmbeddingConfig(**values))
    ids = torch.randint(1, config.vocab_size, (8, 32))
    embedding.train()
    assert not torch.equal(embedding(ids), embedding(ids))
    embedding.eval()
    assert torch.equal(embedding(ids), embedding(ids))


def test_bfloat16_module_controls_output_dtype(config):
    embedding = TokenEmbedding(config).to(torch.bfloat16)
    output = embedding(torch.randint(1, config.vocab_size, (2, 8)))
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()


def test_synthetic_dataset_contract_passes_embedding():
    data_config = SyntheticRetrievalConfig(
        num_train_examples=8,
        num_val_examples=4,
        block_size=32,
        min_filler_tokens=2,
        max_filler_tokens=6,
        num_keys_per_example=2,
        vocab_filler_size=16,
        num_key_types=4,
        num_value_types=8,
        batch_size=2,
        seed=123,
    )
    loader, _, tokenizer = create_synthetic_retrieval_dataloaders(data_config)
    batch = next(iter(loader))
    embedding = TokenEmbedding(
        EmbeddingConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=16,
            pad_token_id=tokenizer.pad_id,
            max_seq_len=32,
        )
    )
    output = embedding(batch)
    assert output.shape == (2, 32, 16)
    assert batch["input_ids"].max() < tokenizer.vocab_size


def test_state_dict_roundtrip_preserves_lookup_exactly(config):
    first = TokenEmbedding(config).eval()
    second = TokenEmbedding(config).eval()
    second.load_state_dict(first.state_dict())
    ids = torch.randint(0, config.vocab_size, (2, 8))
    torch.testing.assert_close(first(ids), second(ids), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_and_gradient(config):
    embedding = TokenEmbedding(config).cuda()
    ids = torch.randint(1, config.vocab_size, (2, 8), device="cuda")
    embedding(ids).sum().backward()
    assert embedding.weight.grad is not None
    assert torch.isfinite(embedding.weight.grad).all()
