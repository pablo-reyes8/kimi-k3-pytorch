"""Residual-branch contribution metrics."""

from __future__ import annotations

from .reducers import cosine, rms, safe_ratio, scalar


def compute_block_contribution(
    input_state,
    branch_output,
    output_state,
    *,
    prefix: str,
) -> dict[str, float]:
    input_rms = rms(input_state)
    branch_rms = rms(branch_output)
    return {
        f"{prefix}/branch_to_input_rms": safe_ratio(branch_rms, input_rms),
        f"{prefix}/state_change_ratio": safe_ratio(
            rms(output_state.detach() - input_state.detach()), input_rms
        ),
        f"{prefix}/branch_input_cosine": cosine(
            branch_output, input_state
        ),
        f"{prefix}/output_rms": rms(output_state),
        f"{prefix}/output_absmax": scalar(
            output_state.detach().float().abs().max()
        ),
        f"{prefix}/output_zero_fraction": scalar(
            (output_state.detach() == 0).float().mean()
        ),
    }
