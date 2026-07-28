from __future__ import annotations

import torch


def build_mla_diagnostics(
    query: torch.Tensor,
    key: torch.Tensor,
    latent_kv: torch.Tensor,
    gate: torch.Tensor,
    attentions: torch.Tensor,
    key_mask: torch.Tensor,
    query_mask: torch.Tensor,
    cache_elements: int,
    full_kv_width: int,
) -> dict[str, torch.Tensor]:
    probabilities = attentions.float()
    positive = probabilities > 0
    entropy_terms = torch.where(
        positive,
        -probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log(),
        torch.zeros_like(probabilities),
    )
    row_valid = positive.any(dim=-1)
    entropy = entropy_terms.sum(dim=-1)[row_valid]
    maximum = probabilities.max(dim=-1).values[row_valid]
    valid_latent = latent_kv[key_mask]
    latent_norm = valid_latent.float().norm(dim=-1)
    gate_values = gate[query_mask].float()
    valid_query = query[query_mask]
    valid_key = key[key_mask]
    cache_length = latent_kv.shape[1]
    latent_dim = latent_kv.shape[-1]
    return {
        "attention_entropy": entropy.mean(),
        "attention_max_probability": maximum.mean(),
        "gate_mean": gate_values.mean(),
        "gate_min": gate_values.min(),
        "gate_max": gate_values.max(),
        "gate_saturation_low": (gate_values < 0.01).float().mean(),
        "gate_saturation_high": (gate_values > 0.99).float().mean(),
        "latent_norm_mean": latent_norm.mean(),
        "latent_norm_max": latent_norm.max(),
        "query_norm_mean": valid_query.float().norm(dim=-1).mean(),
        "key_norm_mean": valid_key.float().norm(dim=-1).mean(),
        "cache_length": torch.tensor(
            cache_length, dtype=torch.long, device=query.device
        ),
        "cache_elements": torch.tensor(
            cache_elements, dtype=torch.long, device=query.device
        ),
        "compression_ratio": torch.tensor(
            full_kv_width / latent_dim,
            dtype=torch.float32,
            device=query.device,
        ),
    }
