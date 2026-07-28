import math

import pytest
import torch

from training.diagnostics import compute_activation_metrics


def test_every_activation_metric_matches_controlled_sample():
    values = torch.tensor([0.0, 2.0, float("inf"), 8.0])
    metrics = compute_activation_metrics(values, max_elements=3)
    assert metrics.keys() == {
        "activation/rms", "activation/mean", "activation/std",
        "activation/absmax", "activation/zero_fraction",
        "activation/nonfinite_fraction",
    }
    # The non-finite sample is counted, then replaced by zero for safe moments.
    assert metrics == pytest.approx({
        "activation/rms": math.sqrt(4 / 3),
        "activation/mean": 2 / 3,
        "activation/std": math.sqrt(8 / 9),
        "activation/absmax": 2.0,
        "activation/zero_fraction": 2 / 3,
        "activation/nonfinite_fraction": 1 / 3,
    })


def test_activation_sampling_is_bounded_detached_and_validated():
    tensor = torch.tensor([1.0, 2.0, 100.0], requires_grad=True)
    metrics = compute_activation_metrics(tensor, max_elements=2)
    assert metrics["activation/absmax"] == 2
    assert all(isinstance(value, float) for value in metrics.values())
    with pytest.raises(ValueError):
        compute_activation_metrics(tensor, max_elements=0)
