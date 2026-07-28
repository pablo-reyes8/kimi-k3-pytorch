"""Detached text-vision pretraining health metrics."""

from __future__ import annotations

from .reducers import rms, scalar


def compute_vision_metrics(
    vision_outputs,
    multimodal_metadata,
    *,
    prefix: str = "vision",
) -> dict[str, float]:
    if vision_outputs is None:
        return {}
    metrics = {}
    for name in ("images", "videos"):
        output = getattr(vision_outputs, name, None)
        if output is None:
            continue
        hidden = output.last_hidden_state
        metrics[f"{prefix}/{name}_hidden_rms"] = rms(hidden)
        metrics[f"{prefix}/{name}_hidden_absmax"] = scalar(
            hidden.detach().float().abs().max()
        )
        metrics[f"{prefix}/{name}_items"] = float(hidden.shape[0])
    if multimodal_metadata is not None:
        for name in ("image_token_counts", "video_token_counts"):
            values = getattr(multimodal_metadata, name, None)
            if values is not None:
                metrics[f"{prefix}/{name}_total"] = scalar(values.sum())
                metrics[f"{prefix}/{name}_mean"] = scalar(
                    values.float().mean()
                )
    return metrics


__all__ = ["compute_vision_metrics"]
