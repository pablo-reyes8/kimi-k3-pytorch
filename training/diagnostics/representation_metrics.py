"""Low-cost representation-collapse proxies."""

from __future__ import annotations

import torch

from .reducers import rms, scalar


def compute_representation_metrics(
    hidden_states: torch.Tensor,
    *,
    max_tokens: int = 256,
    prefix: str = "representation",
) -> dict[str, float]:
    if hidden_states.ndim < 2:
        raise ValueError("hidden_states must have a feature dimension")
    sample = hidden_states.detach().reshape(-1, hidden_states.shape[-1])[
        :max_tokens
    ].float()
    if sample.numel() == 0:
        return {}
    feature_variance = sample.var(dim=0, unbiased=False)
    centered = sample - sample.mean(dim=-1, keepdim=True)
    if sample.shape[0] > 1:
        normalized = torch.nn.functional.normalize(centered, dim=-1)
        cosine_values = (normalized[:-1] * normalized[1:]).sum(dim=-1)
        cosine_mean = scalar(cosine_values.mean())
        cosine_std = scalar(cosine_values.std(unbiased=False))
    else:
        cosine_mean = cosine_std = 0.0
    return {
        f"{prefix}/rms": rms(sample),
        f"{prefix}/feature_mean_abs": scalar(
            sample.mean(dim=0).abs().mean()
        ),
        f"{prefix}/feature_std_mean": scalar(
            feature_variance.sqrt().mean()
        ),
        f"{prefix}/dead_feature_fraction": scalar(
            (feature_variance <= 1e-12).float().mean()
        ),
        f"{prefix}/token_cosine_mean_sampled": cosine_mean,
        f"{prefix}/token_cosine_std_sampled": cosine_std,
        f"{prefix}/effective_variance_ratio_proxy": scalar(
            feature_variance.mean()
            / feature_variance.max().clamp_min(1e-12)
        ),
    }
