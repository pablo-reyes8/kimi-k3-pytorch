import math

import torch


def build_kda_diagnostics(
    q: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    alpha: torch.Tensor,
    final_state: torch.Tensor,
    output_gate: torch.Tensor,
    attention_mask: torch.Tensor | None,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    valid = (
        torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
        if attention_mask is None
        else attention_mask
    )
    valid_h = valid[:, :, None].expand_as(beta)
    valid_hk = valid[:, :, None, None].expand_as(g)
    selected_alpha = alpha[valid_hk]
    selected_g = g[valid_hk]
    selected_beta = beta[valid_h]
    selected_gate = output_gate[
        valid[:, :, None].expand_as(output_gate)
    ]
    q_error = (q.float().norm(dim=-1) - 1).abs()
    k_error = (k.float().norm(dim=-1) - 1).abs()
    state_norm = final_state.float().flatten(-2).norm(dim=-1)
    return {
        "alpha_min": selected_alpha.min(),
        "alpha_max": selected_alpha.max(),
        "alpha_mean": selected_alpha.mean(),
        "log_decay_min": selected_g.min(),
        "log_decay_max": selected_g.max(),
        "beta_mean": selected_beta.mean(),
        "beta_saturation_low": (selected_beta < 0.01).float().mean(),
        "beta_saturation_high": (selected_beta > 0.99).float().mean(),
        "state_norm_mean": state_norm.mean(),
        "state_norm_max": state_norm.max(),
        "q_norm_error": q_error[valid_h].mean(),
        "k_norm_error": k_error[valid_h].mean(),
        "output_gate_mean": selected_gate.mean(),
        "output_gate_saturation": (
            (selected_gate < 0.01) | (selected_gate > 0.99)
        ).float().mean(),
        "chunk_count": torch.tensor(
            math.ceil(q.shape[1] / chunk_size),
            dtype=torch.long,
            device=q.device,
        ),
    }

