"""MoonViT, hierarchical, and Swin vision encoder components."""

import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        activation: str = "gelu",
        bias: bool = False,
    ):
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("all projector dimensions must be > 0")
        activations = {"gelu": nn.GELU, "silu": nn.SiLU}
        if activation not in activations:
            raise ValueError("activation must be 'gelu' or 'silu'")
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.activation = activations[activation]()
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"tokens must have shape [B, N, {self.input_dim}], "
                f"got {tuple(tokens.shape)}"
            )
        return self.fc2(self.activation(self.fc1(tokens)))

