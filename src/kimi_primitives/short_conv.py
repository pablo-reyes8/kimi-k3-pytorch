"""Reusable neural-network primitives shared by Kimi attention implementations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .states import ShortConvState


class CausalShortConv1D(nn.Module):
    """Depthwise causal short convolution over canonical ``[B,T,C]``.

    Parameters are stored in lag order: ``weight[c, j]`` multiplies
    ``x[b, t-j, c]``. Incremental state stores the last ``kernel_size-1``
    input vectors and is never mutated in place.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        self.channels = channels
        self.kernel_size = kernel_size
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(channels, kernel_size, **factory_kwargs)
        )
        self.bias = (
            nn.Parameter(torch.empty(channels, **factory_kwargs)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _validate_state(
        self, x: torch.Tensor, state: ShortConvState | None
    ) -> torch.Tensor:
        batch = x.shape[0]
        history_length = self.kernel_size - 1
        if state is None:
            return x.new_zeros(batch, history_length, self.channels)
        expected = (batch, history_length, self.channels)
        if state.buffer.shape != expected:
            raise ValueError(
                f"state buffer must have shape {expected}, "
                f"got {tuple(state.buffer.shape)}"
            )
        if state.buffer.device != x.device:
            raise ValueError("state buffer and input must be on the same device")
        if state.buffer.dtype != x.dtype:
            raise ValueError("state buffer and input must have the same dtype")
        return state.buffer

    def forward(
        self,
        x: torch.Tensor,
        state: ShortConvState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ShortConvState]:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(
                f"x must have shape [B,T,{self.channels}], got {tuple(x.shape)}"
            )
        history = self._validate_state(x, state)
        combined = torch.cat((history, x), dim=1)
        if x.shape[1] == 0:
            output = x.clone()
        else:
            # conv1d is cross-correlation, hence flip lag-ordered parameters.
            output = F.conv1d(
                combined.transpose(1, 2),
                self.weight.flip(-1).unsqueeze(1),
                self.bias,
                groups=self.channels,
            ).transpose(1, 2)
        if not return_state:
            return output
        history_length = self.kernel_size - 1
        if history_length == 0:
            next_buffer = combined[:, :0]
        else:
            next_buffer = combined[:, -history_length:]
        return output, ShortConvState(next_buffer.clone())

