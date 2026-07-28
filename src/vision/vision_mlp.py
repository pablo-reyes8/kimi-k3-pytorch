import torch
import torch.nn as nn


class VisionMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dim = dim
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")
        return self.dropout2(
            self.fc2(self.dropout1(self.activation(self.fc1(x))))
        )

