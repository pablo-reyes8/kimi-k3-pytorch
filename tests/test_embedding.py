import pytest
import torch

from src.transformer_modules import EmbeddingConfig, TokenEmbedding


def test_embedding_shape_tying_surface_and_validation():
    embedding = TokenEmbedding(EmbeddingConfig(vocab_size=20, d_model=8, max_seq_len=4))
    output = embedding(torch.tensor([[1, 2, 3]]))
    assert output.shape == (1, 3, 8)
    assert embedding.weight is embedding.token_embedding.weight
    with pytest.raises(ValueError):
        embedding(torch.tensor([[20]]))
