"""Download-free Phase 0 smoke training."""

from data import SyntheticRetrievalConfig, create_synthetic_retrieval_dataloaders
from src import BaselineCausalLM, BaselineCausalLMConfig
from training import build_adamw_optimizer, train_one_epoch


def main() -> None:
    data_config = SyntheticRetrievalConfig(
        num_train_examples=32,
        num_val_examples=8,
        block_size=32,
        min_filler_tokens=4,
        max_filler_tokens=12,
        num_keys_per_example=2,
        num_key_types=8,
        num_value_types=16,
        vocab_filler_size=32,
        batch_size=4,
    )
    train_loader, _, tokenizer = create_synthetic_retrieval_dataloaders(data_config)
    model = BaselineCausalLM(
        BaselineCausalLMConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=4,
            mlp_hidden_dim=128,
            max_seq_len=32,
            pad_token_id=tokenizer.pad_id,
        )
    )
    optimizer, _ = build_adamw_optimizer(model, learning_rate=1e-3)
    stats = train_one_epoch(
        model, train_loader, optimizer, device="cpu", max_batches=8
    )
    print(stats)


if __name__ == "__main__":
    main()
