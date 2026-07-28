import torch

from data import SyntheticRetrievalConfig, create_synthetic_retrieval_dataloaders


def test_synthetic_retrieval_batch_is_download_free_and_keeps_answer():
    config = SyntheticRetrievalConfig(
        num_train_examples=8,
        num_val_examples=4,
        block_size=32,
        min_filler_tokens=4,
        max_filler_tokens=8,
        num_keys_per_example=2,
        num_key_types=4,
        num_value_types=8,
        vocab_filler_size=16,
        batch_size=2,
    )
    train, _, tokenizer = create_synthetic_retrieval_dataloaders(config)
    batch = next(iter(train))
    assert set(batch) == {"input_ids", "labels", "attention_mask"}
    assert batch["input_ids"].shape == batch["labels"].shape == (2, 32)
    assert batch["input_ids"].dtype == torch.long
    decoded = tokenizer.decode(batch["input_ids"][0].tolist())
    assert "answer" in decoded
