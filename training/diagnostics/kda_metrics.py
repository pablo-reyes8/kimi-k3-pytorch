"""KDA gate, decay, state and output metrics."""

from __future__ import annotations

from .reducers import scalar


def compute_kda_metrics(diagnostics: dict, prefix: str = "kda") -> dict[str, float]:
    if not diagnostics:
        return {}
    mapping = {
        "alpha_mean": "alpha_mean",
        "alpha_std": "alpha_std",
        "alpha_min_sampled": "alpha_min",
        "alpha_max_sampled": "alpha_max",
        "fraction_alpha_near_lower_bound": "fraction_alpha_near_lower_bound",
        "fraction_alpha_near_one": "fraction_alpha_near_one",
        "beta_mean": "beta_mean",
        "beta_std": "beta_std",
        "fraction_beta_near_zero": "beta_saturation_low",
        "fraction_beta_near_one": "beta_saturation_high",
        "log_decay_mean": "log_decay_mean",
        "cumulative_log_decay_min": "cumulative_log_decay_min",
        "cumulative_log_decay_mean": "cumulative_log_decay_mean",
        "state_rms": "state_rms",
        "state_absmax": "state_absmax",
        "recurrent_output_rms": "recurrent_output_rms",
        "gated_output_rms": "gated_output_rms",
        "output_gate_mean": "output_gate_mean",
        "output_gate_saturation_low": "output_gate_saturation_low",
        "output_gate_saturation_high": "output_gate_saturation_high",
    }
    return {
        f"{prefix}/{public}": scalar(diagnostics[source])
        for public, source in mapping.items()
        if source in diagnostics
    }
