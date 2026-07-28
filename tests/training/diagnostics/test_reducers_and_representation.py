import math

import pytest
import torch

from training.diagnostics import (
    compute_representation_metrics,
    cosine,
    normalized_entropy,
    rms,
    safe_ratio,
    scalar,
    tensor_stats,
)


def test_reducers_measure_known_tensors_and_detach_outputs():
    tensor = torch.tensor([0.0, 2.0], requires_grad=True)
    assert scalar(torch.tensor(3.0, requires_grad=True)) == 3.0
    assert math.isnan(scalar(torch.ones(2)))
    assert rms(tensor) == pytest.approx(math.sqrt(2.0))
    assert safe_ratio(2, -4) == pytest.approx(0.5)
    assert cosine(torch.tensor([1.0, 0]), torch.tensor([0.0, 1])) == 0.0
    assert normalized_entropy(torch.tensor([[0.5, 0.5]])) == pytest.approx(1)
    assert normalized_entropy(torch.tensor([[1.0]])) == 0.0
    assert tensor_stats(tensor, "x") == pytest.approx(
        {
            "x/mean": 1.0, "x/std": 1.0, "x/min": 0.0, "x/max": 2.0,
            "x/rms": math.sqrt(2), "x/absmax": 2.0,
            "x/zero_fraction": 0.5,
        }
    )


def test_every_representation_metric_matches_controlled_features():
    hidden = torch.tensor([[1.0, 0.0], [1.0, 2.0]], requires_grad=True)
    metrics = compute_representation_metrics(hidden)
    assert metrics.keys() == {
        "representation/rms", "representation/feature_mean_abs",
        "representation/feature_std_mean",
        "representation/dead_feature_fraction",
        "representation/token_cosine_mean_sampled",
        "representation/token_cosine_std_sampled",
        "representation/effective_variance_ratio_proxy",
    }
    assert metrics == pytest.approx(
        {
            "representation/rms": math.sqrt(1.5),
            "representation/feature_mean_abs": 1.0,
            "representation/feature_std_mean": 0.5,
            "representation/dead_feature_fraction": 0.5,
            "representation/token_cosine_mean_sampled": -1.0,
            "representation/token_cosine_std_sampled": 0.0,
            "representation/effective_variance_ratio_proxy": 0.5,
        }
    )
    assert all(isinstance(value, float) for value in metrics.values())


def test_representation_token_sampling_is_bounded_and_empty_is_safe():
    hidden = torch.arange(40.0).reshape(10, 4)
    sampled = compute_representation_metrics(hidden, max_tokens=2)
    reference = compute_representation_metrics(hidden[:2], max_tokens=99)
    assert sampled == pytest.approx(reference)
    assert compute_representation_metrics(torch.empty(0, 4)) == {}
