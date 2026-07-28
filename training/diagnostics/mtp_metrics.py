"""MTP objective and representation diagnostics."""

from __future__ import annotations

from .reducers import scalar


def compute_mtp_metrics(
    diagnostics,
    *,
    mtp_loss: float | None = None,
    loss_weight: float = 0.0,
    prefix: str = "mtp",
) -> dict[str, float]:
    if diagnostics is None:
        return {}
    metrics = {
        f"{prefix}/valid_tokens": scalar(diagnostics.valid_token_count),
        f"{prefix}/token_accuracy": scalar(diagnostics.token_accuracy),
        f"{prefix}/logit_entropy": scalar(
            diagnostics.mean_logit_entropy
        ),
        f"{prefix}/hidden_rms": scalar(diagnostics.hidden_rms),
        f"{prefix}/hidden_norm_mean": scalar(
            diagnostics.mean_hidden_norm
        ),
        f"{prefix}/fusion_output_rms": scalar(
            diagnostics.fusion_output_rms
        ),
        f"{prefix}/loss_weight": float(loss_weight),
    }
    block = diagnostics.block or {}
    layer_ratios = []
    for layer in block.get("layers", ()):
        input_norm = layer.get(
            "input_norm", layer.get("pre_attention_depth_norm")
        )
        branch_norm = layer.get("attention_output_norm")
        if input_norm is not None and branch_norm is not None:
            layer_ratios.append(
                scalar(branch_norm) / max(abs(scalar(input_norm)), 1e-12)
            )
    if layer_ratios:
        metrics[f"{prefix}/block_branch_to_input_rms"] = (
            sum(layer_ratios) / len(layer_ratios)
        )
    if mtp_loss is not None:
        metrics[f"{prefix}/loss"] = float(mtp_loss)
    return metrics
