import math
from types import SimpleNamespace

import pytest
import torch

from training.diagnostics import ParameterUpdateMonitor


def test_every_sampled_optimizer_metric_matches_manual_update():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    spec = SimpleNamespace(
        parameter=parameter, parameter_name="token_embedding.weight",
        role="embedding",
    )
    monitor = ParameterUpdateMonitor(
        [spec], max_parameters_per_group=1,
        max_elements_per_parameter=2,
    )
    before = monitor.capture_before_step()
    assert before == pytest.approx(
        {
            "optimizer/embeddings/grad_rms": math.sqrt(12.5),
            "optimizer/embeddings/zero_grad_fraction": 0.0,
            "optimizer/embeddings/nonfinite_grad_count": 0.0,
            "optimizer/embeddings/grad_none_tensors": 0.0,
            "train/gradient_rms_global_sampled": math.sqrt(12.5),
        }
    )
    with torch.no_grad():
        parameter.copy_(torch.tensor([2.0, 4.0]))
    after = monitor.capture_after_step()
    assert after.keys() == {
        "optimizer/embeddings/parameter_rms",
        "optimizer/embeddings/update_rms",
        "optimizer/embeddings/update_to_parameter_ratio",
        "train/parameter_norm_global_sampled",
        "train/update_norm_global_sampled",
        "train/update_to_parameter_ratio_sampled",
        "diagnostics/persistent_gpu_bytes",
    }
    assert after == pytest.approx(
        {
            "optimizer/embeddings/parameter_rms": math.sqrt(10),
            "optimizer/embeddings/update_rms": math.sqrt(2.5),
            "optimizer/embeddings/update_to_parameter_ratio": 0.5,
            "train/parameter_norm_global_sampled": math.sqrt(10),
            "train/update_norm_global_sampled": math.sqrt(2.5),
            "train/update_to_parameter_ratio_sampled": 0.5,
            "diagnostics/persistent_gpu_bytes": 8.0,
        }
    )
    assert monitor.before == {}
    assert monitor.sampled_specs == []
    assert monitor.persistent_bytes == 0


def test_optimizer_metric_sampling_is_deterministic_and_bounded():
    specs = []
    for index in range(4):
        parameter = torch.nn.Parameter(torch.ones(100))
        parameter.grad = torch.ones_like(parameter)
        specs.append(SimpleNamespace(
            parameter=parameter,
            parameter_name=f"embedding_{index}.weight",
            role="embedding",
        ))
    monitor = ParameterUpdateMonitor(
        specs, max_parameters_per_group=2, max_elements_per_parameter=3
    )
    monitor.capture_before_step()
    assert [spec.parameter_name for spec in monitor.sampled_specs] == [
        "embedding_0.weight", "embedding_1.weight"
    ]
    assert monitor.persistent_bytes == 2 * 3 * 4


def test_mtp_gradient_and_update_metrics_have_manual_ratios():
    main = torch.nn.Parameter(torch.tensor([2.0, 2.0]))
    mtp = torch.nn.Parameter(torch.tensor([4.0, 4.0]))
    main.grad = torch.tensor([2.0, 2.0])
    mtp.grad = torch.tensor([1.0, 1.0])
    specs = [
        SimpleNamespace(
            parameter=main, parameter_name="body.weight",
            role="dense_matrix",
        ),
        SimpleNamespace(
            parameter=mtp, parameter_name="model.mtp.projection.weight",
            role="mtp_matrix",
        ),
    ]
    monitor = ParameterUpdateMonitor(specs)
    before = monitor.capture_before_step()
    assert before["mtp/gradient_rms"] == pytest.approx(1.0)
    assert before["mtp/main_to_mtp_grad_ratio"] == pytest.approx(2.0)
    with torch.no_grad():
        main.sub_(0.2)
        mtp.sub_(0.4)
    after = monitor.capture_after_step()
    assert after["mtp/update_rms"] == pytest.approx(0.4)
    assert after["mtp/update_to_parameter_ratio"] == pytest.approx(
        0.4 / 3.6
    )
