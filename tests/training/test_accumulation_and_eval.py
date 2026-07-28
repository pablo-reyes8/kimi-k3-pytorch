from dataclasses import dataclass

import pytest
import torch

from training import TrainerState, eval_one_epoch, train_one_epoch


@dataclass
class Output:
    logits: torch.Tensor
    loss: torch.Tensor


class MaskedScalarLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.75))

    def forward(self, input_ids, labels=None, attention_mask=None):
        values = (self.weight * input_ids.float() - labels.float()).square()
        valid = (
            torch.ones_like(values, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.bool()
        )
        loss = values[valid].mean()
        logits = self.weight * torch.ones(*input_ids.shape, 3)
        return Output(logits=logits, loss=loss)


def variable_token_batches():
    return [
        {
            "input_ids": torch.tensor([[1.0, 2.0, 9.0]]),
            "labels": torch.tensor([[0.0, 1.0, 0.0]]),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        },
        {
            "input_ids": torch.tensor([[3.0, 4.0, 5.0]]),
            "labels": torch.tensor([[1.0, 1.0, 2.0]]),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
        },
    ]


def test_token_weighted_accumulation_matches_concatenated_large_batch():
    accumulated = MaskedScalarLM()
    reference = MaskedScalarLM()
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.03)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)

    train_one_epoch(
        accumulated,
        variable_token_batches(),
        accumulated_optimizer,
        grad_accum_steps=2,
    )
    batches = variable_token_batches()
    large_batch = {
        name: torch.cat([batch[name] for batch in batches], dim=0)
        for name in batches[0]
    }
    train_one_epoch(reference, [large_batch], reference_optimizer)
    torch.testing.assert_close(accumulated.weight, reference.weight)


def test_partial_window_steps_once_and_updates_typed_counters():
    model = MaskedScalarLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    state = TrainerState()
    stats = train_one_epoch(
        model,
        variable_token_batches(),
        optimizer,
        grad_accum_steps=4,
        state=state,
    )
    assert stats["optimizer_steps"] == 1
    assert stats["ntp_tokens"] == 5
    assert state.micro_step == 2
    assert state.optimizer_step == 1
    assert state.tokens_seen == 5
    assert state.samples_seen == 2


def test_eval_is_token_weighted_and_restores_original_training_mode():
    model = MaskedScalarLM().train()
    batches = variable_token_batches()
    with torch.no_grad():
        numerator = denominator = 0.0
        for batch in batches:
            output = model(**batch)
            tokens = float(batch["attention_mask"].sum())
            numerator += float(output.loss) * tokens
            denominator += tokens
    before = model.weight.detach().clone()
    stats = eval_one_epoch(model, batches)
    assert stats["loss"] == pytest.approx(numerator / denominator)
    assert stats["ntp_tokens"] == denominator
    assert model.training
    torch.testing.assert_close(model.weight, before)
    assert model.weight.grad is None


def test_empty_eval_has_explicit_neutral_counts():
    model = MaskedScalarLM().eval()
    stats = eval_one_epoch(model, [])
    assert stats["num_batches"] == 0
    assert stats["ntp_tokens"] == 0
    assert stats["ntp_perplexity"] == 1
    assert not model.training
