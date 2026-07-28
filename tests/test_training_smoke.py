import math

import pytest
import torch
from torch.utils.data import DataLoader

from conftest import tiny_model
from data import SyntheticRetrievalConfig, create_synthetic_retrieval_dataloaders
from src import BaselineCausalLM, BaselineCausalLMConfig, CausalLMOutput
from training import build_adamw_optimizer, eval_one_epoch, train_one_epoch
from training.scheduler import WarmupCosineLR
from training.train_model import train_model


def random_loader(
    *, n_examples=6, batch_size=2, seq_len=12, vocab_size=24, repeated=False
):
    generator = torch.Generator().manual_seed(4)
    ids = torch.randint(0, vocab_size, (n_examples, seq_len), generator=generator)
    labels = torch.roll(ids, shifts=-1, dims=1)
    if repeated:
        ids[:] = ids[0]
        labels[:] = labels[0]
    samples = [
        {"input_ids": input_ids, "labels": target}
        for input_ids, target in zip(ids, labels)
    ]
    return DataLoader(samples, batch_size=batch_size, shuffle=False)


def test_train_updates_parameters_steps_scheduler_and_reports_exact_counts():
    model = tiny_model(vocab_size=24, pad_token_id=None)
    optimizer, _ = build_adamw_optimizer(model, learning_rate=2e-3)
    scheduler = WarmupCosineLR(optimizer, total_steps=3, warmup_steps=1)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    stats = train_one_epoch(
        model,
        random_loader(),
        optimizer,
        scheduler=scheduler,
        grad_clip=1.0,
    )
    assert stats["num_batches"] == 3
    assert stats["optimizer_steps"] == 3
    assert scheduler.step_num == 3
    assert math.isfinite(stats["loss"])
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gradient_accumulation_steps_final_partial_window():
    model = tiny_model(vocab_size=24, pad_token_id=None)
    optimizer, _ = build_adamw_optimizer(model, learning_rate=2e-3)
    stats = train_one_epoch(
        model,
        random_loader(n_examples=6, batch_size=2),
        optimizer,
        grad_accum_steps=2,
    )
    assert stats["num_batches"] == 3
    assert stats["optimizer_steps"] == 2
    assert all(parameter.grad is None for parameter in model.parameters())


def test_single_partial_accumulation_window_has_same_update_as_one_step():
    class ScalarLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, input_ids, labels=None):
            loss = (self.weight * input_ids.float()).mean().square()
            return CausalLMOutput(
                logits=torch.zeros(*input_ids.shape, 2),
                loss=loss,
            )

    loader = DataLoader(
        [{"input_ids": torch.ones(2), "labels": torch.ones(2, dtype=torch.long)}],
        batch_size=1,
    )
    regular, accumulated = ScalarLossModel(), ScalarLossModel()
    regular_optimizer = torch.optim.SGD(regular.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    train_one_epoch(regular, loader, regular_optimizer, grad_accum_steps=1)
    train_one_epoch(accumulated, loader, accumulated_optimizer, grad_accum_steps=4)
    torch.testing.assert_close(regular.weight, accumulated.weight)


@pytest.mark.parametrize("grad_accum_steps", [0, -1])
def test_invalid_gradient_accumulation_rejected(grad_accum_steps):
    model = tiny_model(vocab_size=24)
    optimizer, _ = build_adamw_optimizer(model)
    with pytest.raises(ValueError):
        train_one_epoch(
            model,
            random_loader(),
            optimizer,
            grad_accum_steps=grad_accum_steps,
        )


def test_max_batches_zero_and_two_are_respected():
    for maximum, expected in ((0, 0), (2, 2)):
        model = tiny_model(vocab_size=24, pad_token_id=None)
        optimizer, _ = build_adamw_optimizer(model)
        stats = train_one_epoch(
            model, random_loader(), optimizer, max_batches=maximum
        )
        assert stats["num_batches"] == expected


def test_missing_model_loss_rejected():
    class NoLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.ones(1))

        def forward(self, input_ids, labels=None):
            logits = self.parameter * torch.ones(*input_ids.shape, 2)
            return CausalLMOutput(logits=logits)

    model = NoLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="contain a loss"):
        train_one_epoch(model, random_loader(vocab_size=2), optimizer, max_batches=1)


def test_cpu_bfloat16_autocast_training_is_finite():
    model = tiny_model(vocab_size=24, pad_token_id=None)
    optimizer, _ = build_adamw_optimizer(model)
    stats = train_one_epoch(
        model,
        random_loader(n_examples=2),
        optimizer,
        amp_enabled=True,
        amp_dtype="bf16",
    )
    assert math.isfinite(stats["loss"])


def test_eval_matches_manual_batch_mean_and_does_not_create_gradients():
    model = tiny_model(vocab_size=24, pad_token_id=None)
    loader = random_loader()
    manual = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            manual.append(model(**batch).loss.item())
    stats = eval_one_epoch(model, loader)
    assert stats["loss"] == pytest.approx(sum(manual) / len(manual))
    assert stats["perplexity"] == pytest.approx(math.exp(stats["loss"]))
    assert stats["num_batches"] == 3
    assert all(parameter.grad is None for parameter in model.parameters())


def test_eval_requires_loss_and_respects_max_batches():
    model = tiny_model(vocab_size=24, pad_token_id=None)
    assert eval_one_epoch(model, random_loader(), max_batches=1)["num_batches"] == 1

    class NoLoss(torch.nn.Module):
        def forward(self, input_ids, labels=None):
            return CausalLMOutput(logits=torch.zeros(*input_ids.shape, 2))

    with pytest.raises(ValueError):
        eval_one_epoch(NoLoss(), random_loader(vocab_size=2), max_batches=1)


def test_train_model_orchestrates_train_and_validation_histories():
    model = tiny_model(vocab_size=24, pad_token_id=None)
    optimizer, _ = build_adamw_optimizer(model)
    loader = random_loader(n_examples=4)
    history = train_model(
        model,
        loader,
        optimizer,
        epochs=2,
        val_loader=loader,
        max_batches=1,
    )
    assert len(history["train"]) == len(history["validation"]) == 2
    assert all(item["num_batches"] == 1 for item in history["train"])
    assert all(item["num_batches"] == 1 for item in history["validation"])


def test_tiny_synthetic_retrieval_batch_can_be_overfit_on_cpu():
    config = SyntheticRetrievalConfig(
        num_train_examples=1,
        num_val_examples=1,
        block_size=20,
        min_filler_tokens=0,
        max_filler_tokens=0,
        num_keys_per_example=1,
        vocab_filler_size=4,
        num_key_types=1,
        num_value_types=2,
        batch_size=1,
        seed=9,
    )
    loader, _, tokenizer = create_synthetic_retrieval_dataloaders(config)
    batch = next(iter(loader))
    repeated_loader = DataLoader([batch] * 12, batch_size=None)
    model = BaselineCausalLM(
        BaselineCausalLMConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=24,
            n_layers=1,
            n_heads=3,
            mlp_hidden_dim=48,
            max_seq_len=20,
            pad_token_id=tokenizer.pad_id,
        )
    )
    optimizer, _ = build_adamw_optimizer(
        model, learning_rate=5e-3, weight_decay=0
    )
    model.eval()
    before = model(**batch).loss.item()
    train_one_epoch(model, repeated_loader, optimizer, max_batches=12)
    model.eval()
    after = model(**batch).loss.item()
    assert after < before * 0.7, (before, after)
