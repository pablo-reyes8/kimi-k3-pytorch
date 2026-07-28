import math

import pytest
import torch
import torch.nn.functional as F

from training.training_metrics import compute_lm_metrics


def test_metrics_match_manual_cross_entropy_perplexity_and_accuracy():
    logits = torch.tensor(
        [[[5.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 3.0, 1.0]]]
    )
    labels = torch.tensor([[0, 1, 2]])
    metrics = compute_lm_metrics(logits, labels)
    expected_loss = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))
    assert metrics["loss"] == pytest.approx(expected_loss.item())
    assert metrics["perplexity"] == pytest.approx(math.exp(expected_loss.item()))
    assert metrics["token_accuracy"] == pytest.approx(2 / 3)


def test_ignore_index_excludes_tokens_from_loss_and_accuracy():
    logits = torch.tensor([[[4.0, 0.0], [4.0, 0.0]]])
    labels = torch.tensor([[0, -7]])
    metrics = compute_lm_metrics(logits, labels, ignore_index=-7)
    assert metrics["token_accuracy"] == 1.0
    assert metrics["loss"] == pytest.approx(
        F.cross_entropy(logits[:, :1].reshape(-1, 2), torch.tensor([0])).item()
    )


def test_all_ignored_returns_nan_metrics_not_misleading_zero():
    metrics = compute_lm_metrics(
        torch.randn(2, 3, 5), torch.full((2, 3), -100)
    )
    assert all(math.isnan(value) for value in metrics.values())


def test_large_loss_perplexity_is_capped_to_prevent_overflow():
    logits = torch.tensor([[[1000.0, -1000.0]]])
    metrics = compute_lm_metrics(logits, torch.tensor([[1]]))
    assert metrics["perplexity"] == pytest.approx(math.exp(50))


def test_noncontiguous_logits_and_labels_supported():
    logits = torch.randn(2, 5, 3).transpose(0, 1)
    labels = torch.randint(0, 3, (2, 5)).transpose(0, 1)
    metrics = compute_lm_metrics(logits, labels)
    assert math.isfinite(metrics["loss"])
