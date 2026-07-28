"""Small detached reductions shared by all diagnostic families."""

from __future__ import annotations

import math

import torch


def scalar(value, default: float = float("nan")) -> float:
    try:
        if torch.is_tensor(value):
            if value.numel() != 1:
                return default
            return float(value.detach().float().item())
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return default


def rms(tensor: torch.Tensor) -> float:
    return scalar(tensor.detach().float().square().mean().sqrt())


def safe_ratio(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    return float(numerator / max(abs(denominator), eps))


def cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    if left.shape != right.shape:
        raise ValueError("cosine inputs must have identical shapes")
    value = torch.nn.functional.cosine_similarity(
        left.detach().float().reshape(-1),
        right.detach().float().reshape(-1),
        dim=0,
        eps=eps,
    )
    return scalar(value)


def tensor_stats(tensor: torch.Tensor, prefix: str) -> dict[str, float]:
    values = tensor.detach().float()
    if values.numel() == 0:
        return {}
    return {
        f"{prefix}/mean": scalar(values.mean()),
        f"{prefix}/std": scalar(values.std(unbiased=False)),
        f"{prefix}/min": scalar(values.min()),
        f"{prefix}/max": scalar(values.max()),
        f"{prefix}/rms": rms(values),
        f"{prefix}/absmax": scalar(values.abs().max()),
        f"{prefix}/zero_fraction": scalar((values == 0).float().mean()),
    }


def normalized_entropy(weights: torch.Tensor, eps: float = 1e-12) -> float:
    probabilities = weights.detach().float()
    probabilities = probabilities / probabilities.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    entropy = -(
        probabilities
        * probabilities.clamp_min(eps).log()
    ).sum(dim=-1)
    support = probabilities.shape[-1]
    if support <= 1:
        return 0.0
    return scalar(entropy.mean() / math.log(support))


def ensure_plain_scalars(metrics: dict[str, object]) -> dict[str, float]:
    result = {}
    for name, value in metrics.items():
        converted = scalar(value)
        if not math.isnan(converted) or (
            torch.is_tensor(value) and value.numel() == 1
        ):
            result[name] = converted
    return result
