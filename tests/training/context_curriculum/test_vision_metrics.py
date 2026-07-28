from types import SimpleNamespace

import pytest
import torch

from training.diagnostics import compute_vision_metrics


def test_every_vision_metric_matches_controlled_multimodal_outputs():
    outputs = SimpleNamespace(
        images=SimpleNamespace(
            last_hidden_state=torch.tensor([[[3.0, 4.0]]])
        ),
        videos=SimpleNamespace(
            last_hidden_state=torch.tensor([
                [[0.0, 2.0]], [[0.0, 2.0]]
            ])
        ),
    )
    metadata = SimpleNamespace(
        image_token_counts=torch.tensor([4, 8]),
        video_token_counts=torch.tensor([6, 2]),
    )
    metrics = compute_vision_metrics(outputs, metadata)
    assert metrics.keys() == {
        "vision/images_hidden_rms", "vision/images_hidden_absmax",
        "vision/images_items", "vision/videos_hidden_rms",
        "vision/videos_hidden_absmax", "vision/videos_items",
        "vision/image_token_counts_total",
        "vision/image_token_counts_mean",
        "vision/video_token_counts_total",
        "vision/video_token_counts_mean",
    }
    assert metrics == pytest.approx({
        "vision/images_hidden_rms": (12.5) ** 0.5,
        "vision/images_hidden_absmax": 4.0,
        "vision/images_items": 1.0,
        "vision/videos_hidden_rms": 2 ** 0.5,
        "vision/videos_hidden_absmax": 2.0,
        "vision/videos_items": 2.0,
        "vision/image_token_counts_total": 12.0,
        "vision/image_token_counts_mean": 6.0,
        "vision/video_token_counts_total": 8.0,
        "vision/video_token_counts_mean": 4.0,
    })
    assert all(isinstance(value, float) for value in metrics.values())


def test_text_only_output_emits_no_fake_vision_metrics():
    assert compute_vision_metrics(None, None) == {}
