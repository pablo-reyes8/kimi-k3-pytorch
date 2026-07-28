from __future__ import annotations

import torch


def validate_sources(
    sources: torch.Tensor, d_model: int
) -> tuple[int, int, int]:
    if sources.ndim != 4 or sources.shape[-1] != d_model:
        raise ValueError(
            f"sources must have shape [B,T,S,{d_model}], "
            f"got {tuple(sources.shape)}"
        )
    if sources.shape[2] == 0:
        raise ValueError("AttnRes requires at least one depth source")
    if not sources.dtype.is_floating_point:
        raise TypeError("sources must be floating point")
    return sources.shape[:3]


def accumulation_dtype(
    dtype: torch.dtype, enabled: bool
) -> torch.dtype:
    if enabled and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def stack_sources(sources: list[torch.Tensor]) -> torch.Tensor:
    if not sources:
        raise ValueError("source list must not be empty")
    shape = sources[0].shape
    if any(source.shape != shape for source in sources):
        raise ValueError("all depth sources must have identical [B,T,D] shapes")
    if any(source.device != sources[0].device for source in sources):
        raise ValueError("all depth sources must share device")
    if any(source.dtype != sources[0].dtype for source in sources):
        raise TypeError("all depth sources must share dtype")
    return torch.stack(sources, dim=2)
