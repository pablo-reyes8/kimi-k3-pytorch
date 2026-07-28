"""Head-layout descriptors and independent per-head orthogonalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .newton_schulz import match_update_rms, zeropower_via_newton_schulz


@dataclass(frozen=True)
class HeadMatrixLayout:
    num_heads: int
    head_dim: int
    head_axis: Literal[0, 1]
    input_dim: int
    output_dim: int
    packed_kind: Literal["separate", "fused_qkv"] = "separate"
    qkv_slices: tuple[slice, slice, slice] | None = None

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        if self.head_axis not in (0, 1):
            raise ValueError("head_axis must be 0 or 1")
        if self.packed_kind == "fused_qkv" and self.qkv_slices is None:
            raise ValueError("fused_qkv layout requires explicit slices")

    def validate(self, matrix: torch.Tensor) -> None:
        if matrix.ndim != 2:
            raise ValueError("head matrix must be 2D")
        if tuple(matrix.shape) != (self.output_dim, self.input_dim):
            raise ValueError(
                f"matrix shape {tuple(matrix.shape)} does not match "
                f"layout {(self.output_dim, self.input_dim)}"
            )
        if self.packed_kind == "separate":
            expected = self.num_heads * self.head_dim
            if matrix.shape[self.head_axis] != expected:
                raise ValueError("head axis does not equal num_heads * head_dim")


def split_head_matrix(
    matrix: torch.Tensor, layout: HeadMatrixLayout
) -> tuple[torch.Tensor, ...]:
    layout.validate(matrix)
    if layout.packed_kind != "separate":
        raise ValueError("split_head_matrix expects a separate Q/K/V layout")
    if layout.head_axis == 0:
        shaped = matrix.reshape(
            layout.num_heads, layout.head_dim, matrix.shape[1]
        )
    else:
        shaped = matrix.reshape(
            matrix.shape[0], layout.num_heads, layout.head_dim
        ).permute(1, 0, 2)
    return tuple(shaped[index] for index in range(layout.num_heads))


def merge_head_matrix(
    blocks: tuple[torch.Tensor, ...] | list[torch.Tensor],
    layout: HeadMatrixLayout,
) -> torch.Tensor:
    if len(blocks) != layout.num_heads:
        raise ValueError("number of blocks does not match num_heads")
    stacked = torch.stack(tuple(blocks))
    if layout.head_axis == 0:
        merged = stacked.reshape(layout.output_dim, layout.input_dim)
    else:
        merged = stacked.permute(1, 0, 2).reshape(
            layout.output_dim, layout.input_dim
        )
    return merged


@torch.no_grad()
def per_head_orthogonalize(
    matrix: torch.Tensor,
    layout: HeadMatrixLayout,
    *,
    steps: int,
    eps: float,
    rms_scaling: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    layout.validate(matrix)
    raw_rms: list[float] = []
    orthogonal_rms: list[float] = []
    final_rms: list[float] = []

    def transform_separate(part, child_layout):
        transformed = []
        for block in split_head_matrix(part, child_layout):
            raw_rms.append(
                float(block.float().square().mean().sqrt().item())
            )
            update = zeropower_via_newton_schulz(
                block, steps=steps, eps=eps
            )
            orthogonal_rms.append(
                float(update.float().square().mean().sqrt().item())
            )
            if rms_scaling:
                update = match_update_rms(update, mode="shape")
            transformed.append(update)
            final_rms.append(
                float(update.float().square().mean().sqrt().item())
            )
        return merge_head_matrix(transformed, child_layout)

    if layout.packed_kind == "fused_qkv":
        output = torch.zeros_like(matrix)
        for packed_slice in layout.qkv_slices:
            part = (
                matrix[packed_slice, :]
                if layout.head_axis == 0
                else matrix[:, packed_slice]
            )
            child = HeadMatrixLayout(
                num_heads=layout.num_heads,
                head_dim=layout.head_dim,
                head_axis=layout.head_axis,
                input_dim=part.shape[1],
                output_dim=part.shape[0],
            )
            transformed = transform_separate(part, child)
            if layout.head_axis == 0:
                output[packed_slice, :] = transformed
            else:
                output[:, packed_slice] = transformed
    else:
        output = transform_separate(matrix, layout)

    def summary(values, label):
        tensor = torch.tensor(values, dtype=torch.float64)
        mean = float(tensor.mean().item())
        std = float(tensor.std(unbiased=False).item())
        return {
            f"per_head_muon/{label}_mean": mean,
            f"per_head_muon/{label}_std": std,
            f"per_head_muon/{label}_min": float(tensor.min().item()),
            f"per_head_muon/{label}_max": float(tensor.max().item()),
            f"per_head_muon/{label}_median": float(tensor.median().item()),
            f"per_head_muon/{label}_cv": std / max(mean, eps),
        }

    metrics = {}
    metrics.update(summary(raw_rms, "raw_momentum_rms"))
    metrics.update(summary(orthogonal_rms, "orthogonal_update_rms"))
    metrics.update(summary(final_rms, "head_update_rms"))
    final_values = torch.tensor(final_rms, dtype=torch.float64)
    metrics["per_head_muon/head_update_rms_max_to_median"] = float(
        final_values.max().item()
    ) / max(float(final_values.median().item()), eps)
    metrics["per_head_muon/fraction_heads_near_zero_update"] = float(
        (final_values <= eps).double().mean().item()
    )
    return output, metrics
