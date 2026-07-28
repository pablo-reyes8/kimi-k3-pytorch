import torch
from torch.utils.data import DataLoader

from conftest import tiny_model
from training import build_adamw_optimizer, train_one_epoch


def test_cpu_training_smoke_and_tiny_overfit():
    torch.manual_seed(4)
    model = tiny_model(vocab_size=24, pad_token_id=None)
    input_ids = torch.randint(0, 24, (4, 12))
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loader = DataLoader(
        [{"input_ids": input_ids[0], "labels": labels[0]}] * 8,
        batch_size=2,
        shuffle=False,
    )
    optimizer, _ = build_adamw_optimizer(model, learning_rate=3e-3, weight_decay=0.0)
    before = model(input_ids[:1], labels=labels[:1]).loss.item()
    stats = train_one_epoch(model, loader, optimizer, max_batches=4)
    after = model(input_ids[:1], labels=labels[:1]).loss.item()
    assert stats["optimizer_steps"] == 4
    assert after < before
