"""Reusable neural-network primitives shared by Kimi attention implementations."""

from __future__ import annotations

import torch
import torch.nn as nn


class FullRankOutputGate(nn.Module):
    """Input-conditioned full-rank gate followed by an output projection.

    ``output_proj(sigmoid(gate_proj(residual_input)) * attention_output)``.
    Residual addition intentionally remains the caller's responsibility.
    """

    def __init__(
        self,
        d_model: int,
        bias: bool = False,
        output_bias: bool = False,
        init_std: float = 0.02,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if init_std <= 0:
            raise ValueError("init_std must be > 0")
        self.d_model = d_model
        self.init_std = float(init_std)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(
            d_model, d_model, bias=bias, **factory_kwargs
        )
        self.output_proj = nn.Linear(
            d_model, d_model, bias=output_bias, **factory_kwargs
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (self.gate_proj, self.output_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=self.init_std)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def gate_values(self, residual_input: torch.Tensor) -> torch.Tensor:
        if residual_input.ndim != 3 or residual_input.shape[-1] != self.d_model:
            raise ValueError(
                f"residual_input must have shape [B,T,{self.d_model}], "
                f"got {tuple(residual_input.shape)}"
            )
        return torch.sigmoid(self.gate_proj(residual_input))

    def forward(
        self,
        attention_output: torch.Tensor,
        residual_input: torch.Tensor,
        *,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if attention_output.ndim != 3 or attention_output.shape[-1] != self.d_model:
            raise ValueError(
                f"attention_output must have shape [B,T,{self.d_model}], "
                f"got {tuple(attention_output.shape)}"
            )
        if attention_output.shape != residual_input.shape:
            raise ValueError(
                "attention_output and residual_input must have identical shapes"
            )
        gate = self.gate_values(residual_input)
        output = self.output_proj(gate * attention_output)
        return (output, gate) if return_gate else output

