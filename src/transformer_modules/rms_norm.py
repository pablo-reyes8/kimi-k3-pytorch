import torch 
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Input:
        x: [..., D]

    Forward:
        1. Save original dtype.
        2. Cast x to float32 for stable RMS computation.
        3. Compute mean(x^2) over last dimension.
        4. Apply rsqrt(mean_square + eps).
        5. Normalize x.
        6. Cast normalized x back to original dtype if needed.
        7. Multiply by learnable weight.
        8. Return y.

    Output:
        y: [..., D]
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")

        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      if x.shape[-1] != self.dim:
          raise ValueError(
              f"Expected last dimension to be dim={self.dim}, "
              f"but got x.shape[-1]={x.shape[-1]}"
          )

      original_dtype = x.dtype

      # Compute RMS in float32 for numerical stability
      x_float = x.float()

      mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)
      inv_rms = torch.rsqrt(mean_square + self.eps)

      y = x_float * inv_rms

      # Cast normalized activations back to original dtype
      if y.dtype != original_dtype:
          y = y.to(original_dtype)

      # Important: cast weight too, otherwise output is promoted to float32
      weight = self.weight.to(dtype=original_dtype)

      y = y * weight

      return y