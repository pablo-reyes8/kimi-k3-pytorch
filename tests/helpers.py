import torch

from src import BaselineCausalLM, BaselineCausalLMConfig


def tiny_model(vocab_size: int = 48, pad_token_id: int | None = 0):
    torch.manual_seed(0)
    return BaselineCausalLM(
        BaselineCausalLMConfig(
            vocab_size=vocab_size,
            d_model=32,
            n_layers=2,
            n_heads=4,
            mlp_hidden_dim=64,
            max_seq_len=32,
            pad_token_id=pad_token_id,
            dropout=0.0,
        )
    )
