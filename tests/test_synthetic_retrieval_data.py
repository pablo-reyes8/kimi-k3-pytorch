import pytest
import torch

from data.synthetic_long_context_retrieval import (
    SimpleWordTokenizer,
    SyntheticRetrievalConfig,
    SyntheticRetrievalDataset,
    SyntheticRetrievalGenerator,
    SyntheticRetrievalMTPDataset,
    create_synthetic_retrieval_dataloaders,
)


def make_config(**overrides):
    values = dict(
        num_train_examples=8,
        num_val_examples=4,
        block_size=32,
        min_filler_tokens=2,
        max_filler_tokens=8,
        num_keys_per_example=2,
        vocab_filler_size=16,
        num_key_types=4,
        num_value_types=8,
        batch_size=2,
        num_workers=0,
        seed=123,
    )
    values.update(overrides)
    return SyntheticRetrievalConfig(**values)


@pytest.mark.parametrize(
    "override",
    [
        {"num_train_examples": 0},
        {"num_val_examples": 0},
        {"block_size": 0},
        {"batch_size": 0},
        {"num_keys_per_example": 0},
        {"num_keys_per_example": 5},
        {"min_filler_tokens": -1},
        {"min_filler_tokens": 4, "max_filler_tokens": 3},
    ],
)
def test_invalid_configuration_rejected(override):
    with pytest.raises(ValueError):
        make_config(**override).validate()


def test_tokenizer_vocab_is_complete_unique_and_roundtrips_known_tokens():
    config = make_config()
    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(config)
    expected = 4 + 16 + config.num_key_types + config.num_value_types + config.vocab_filler_size
    assert tokenizer.vocab_size == expected
    assert len(tokenizer.token_to_idx) == len(tokenizer.idx_to_token)
    assert (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id) == (0, 1, 2)
    text = "<bos> key_1 is value_2 filler_3 <eos>"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode("not_in_vocab") == [tokenizer.token_to_idx["<unk>"]]
    before = tokenizer.vocab_size
    tokenizer.add_token("key_1")
    assert tokenizer.vocab_size == before


def test_generators_with_equal_seed_are_reproducible_and_metadata_is_consistent():
    config = make_config()
    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(config)
    first = SyntheticRetrievalGenerator(config, tokenizer, seed=5)
    second = SyntheticRetrievalGenerator(config, tokenizer, seed=5)
    text_a, meta_a = first.generate_text_example()
    text_b, meta_b = second.generate_text_example()
    assert text_a == text_b and meta_a == meta_b
    assert meta_a["query_key"] in meta_a["kv_pairs"]
    assert meta_a["answer_value"] == meta_a["kv_pairs"][meta_a["query_key"]]
    assert f"answer : {meta_a['answer_value']}" in text_a
    assert config.min_filler_tokens <= meta_a["filler_len"] <= config.max_filler_tokens


def test_dataset_length_shapes_shift_mask_and_answer_retention():
    config = make_config()
    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(config)
    dataset = SyntheticRetrievalDataset(config, tokenizer, split="train")
    assert len(dataset) == config.num_train_examples
    sample = dataset[0]
    assert set(sample) == {"input_ids", "labels", "attention_mask"}
    assert sample["input_ids"].shape == sample["labels"].shape == (config.block_size,)
    assert sample["attention_mask"].dtype == torch.bool
    torch.testing.assert_close(sample["input_ids"][1:], sample["labels"][:-1])
    assert torch.equal(sample["attention_mask"], sample["input_ids"].ne(tokenizer.pad_id))
    decoded = tokenizer.decode(sample["input_ids"].tolist())
    assert "question" in decoded and "answer" in decoded


def test_dataset_index_is_deterministic_and_train_validation_are_distinct():
    config = make_config()
    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(config)
    train = SyntheticRetrievalDataset(config, tokenizer, split="train")
    validation = SyntheticRetrievalDataset(config, tokenizer, split="val")
    torch.testing.assert_close(train[3]["input_ids"], train[3]["input_ids"])
    assert not torch.equal(train[0]["input_ids"], validation[0]["input_ids"])


def test_invalid_split_rejected():
    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(make_config())
    with pytest.raises(AssertionError):
        SyntheticRetrievalDataset(make_config(), tokenizer, split="test")


def test_mtp_dataset_offsets_are_exact():
    config = make_config()
    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(config)
    sample = SyntheticRetrievalMTPDataset(
        config, tokenizer, split="train", mtp_depth=3
    )[0]
    assert sample["mtp_labels"].shape == (3, config.block_size)
    torch.testing.assert_close(sample["input_ids"][1:], sample["labels"][:-1])
    torch.testing.assert_close(sample["labels"][1:], sample["mtp_labels"][0, :-1])
    torch.testing.assert_close(
        sample["mtp_labels"][0, 1:], sample["mtp_labels"][1, :-1]
    )


@pytest.mark.parametrize("use_mtp", [False, True])
def test_dataloader_factory_batch_contract_and_split_sizes(use_mtp):
    config = make_config()
    train, validation, tokenizer = create_synthetic_retrieval_dataloaders(
        config, use_mtp=use_mtp, mtp_depth=2
    )
    train_batch = next(iter(train))
    validation_batch = next(iter(validation))
    assert train_batch["input_ids"].shape == (2, 32)
    assert validation_batch["input_ids"].shape == (2, 32)
    assert len(train.dataset) == 8 and len(validation.dataset) == 4
    assert train.dataset.tokenizer is tokenizer
    assert ("mtp_labels" in train_batch) is use_mtp


def test_tiny_block_size_still_keeps_suffix_when_structurally_possible():
    config = make_config(
        block_size=20,
        num_keys_per_example=1,
        min_filler_tokens=100,
        max_filler_tokens=100,
    )
    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(config)
    sample = SyntheticRetrievalDataset(config, tokenizer)[0]
    decoded = tokenizer.decode(sample["input_ids"].tolist())
    assert "answer" in decoded
