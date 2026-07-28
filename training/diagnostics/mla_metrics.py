"""Gated MLA attention, gate and Q/K scale metrics."""

from __future__ import annotations

from .reducers import scalar


def compute_mla_metrics(
    diagnostics: dict,
    *,
    qk_clip_active: bool = False,
    prefix: str = "mla",
) -> dict[str, float]:
    if not diagnostics:
        return {}
    mapping = {
        "attention_entropy_normalized_sampled": "attention_entropy_normalized",
        "max_attention_probability_sampled": "attention_max_probability",
        "output_rms": "attention_output_rms",
        "gated_output_rms": "gated_output_rms",
        "output_gate_mean": "gate_mean",
        "output_gate_saturation_low": "gate_saturation_low",
        "output_gate_saturation_high": "gate_saturation_high",
        "q_rms": "q_rms",
        "k_rms": "k_rms",
        "v_rms": "v_rms",
        "qk_scale_max": "qk_scale_max",
    }
    metrics = {
        f"{prefix}/{public}": scalar(diagnostics[source])
        for public, source in mapping.items()
        if source in diagnostics
    }
    metrics[f"{prefix}/qk_clip_active"] = float(qk_clip_active)
    return metrics
