"""Attention-residual components for mixing hidden-state streams across model depth."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .outputs import DepthSoftmaxStats
from .utils import stack_sources


@dataclass
class BlockAttentionResidualState:
    """Mutable hidden-state history used by block attention residuals."""

    embedding: torch.Tensor
    sublayers_per_depth_block: int
    completed_blocks: list[torch.Tensor] = field(default_factory=list)
    partial_block: torch.Tensor | None = None
    outputs_in_current_block: int = 0
    total_sublayer_outputs: int = 0
    current_depth_block_index: int = 0
    block_sizes: list[int] = field(default_factory=list)
    phase_stats: dict[int, DepthSoftmaxStats] = field(default_factory=dict)
    prepared_depth_block_index: int | None = None
    inter_block_scan_count: int = 0

    def __post_init__(self) -> None:
        if self.embedding.ndim != 3:
            raise ValueError("embedding must have shape [B,T,D]")
        if self.sublayers_per_depth_block <= 0:
            raise ValueError("sublayers_per_depth_block must be > 0")
        if (self.partial_block is None) != (
            self.outputs_in_current_block == 0
        ):
            raise ValueError("partial block/count invariant violated")
        if self.outputs_in_current_block > self.sublayers_per_depth_block:
            raise ValueError("partial block exceeds configured block size")
        stack_sources([self.embedding] + self.completed_blocks)
        if self.partial_block is not None:
            stack_sources([self.embedding, self.partial_block])

    @property
    def source_elements(self) -> int:
        tensors = [self.embedding] + self.completed_blocks
        if self.partial_block is not None:
            tensors.append(self.partial_block)
        return sum(tensor.numel() for tensor in tensors)

    def prepare_for_site(self) -> None:
        if self.outputs_in_current_block == self.sublayers_per_depth_block:
            self.close_current_block()

    def available_sources(self) -> torch.Tensor:
        self.prepare_for_site()
        sources = [self.embedding] + self.completed_blocks
        if self.partial_block is not None:
            sources.append(self.partial_block)
        return stack_sources(sources)

    def completed_sources(self) -> torch.Tensor:
        self.prepare_for_site()
        return stack_sources([self.embedding] + self.completed_blocks)

    def append_output(self, output: torch.Tensor) -> None:
        stack_sources([self.embedding, output])
        if self.outputs_in_current_block == self.sublayers_per_depth_block:
            raise RuntimeError("close full block before appending another output")
        self.partial_block = (
            output
            if self.partial_block is None
            else self.partial_block + output
        )
        self.outputs_in_current_block += 1
        self.total_sublayer_outputs += 1

    def close_current_block(self) -> None:
        if self.partial_block is None:
            return
        self.completed_blocks.append(self.partial_block)
        self.block_sizes.append(self.outputs_in_current_block)
        self.partial_block = None
        self.outputs_in_current_block = 0
        self.current_depth_block_index += 1
        self.phase_stats.clear()
        self.prepared_depth_block_index = None

    def finalize(self) -> None:
        self.close_current_block()

    def clone(self, detach: bool = False) -> "BlockAttentionResidualState":
        copy_tensor = (
            (lambda tensor: tensor.detach().clone())
            if detach
            else (lambda tensor: tensor.clone())
        )
        return BlockAttentionResidualState(
            embedding=copy_tensor(self.embedding),
            sublayers_per_depth_block=self.sublayers_per_depth_block,
            completed_blocks=[
                copy_tensor(block) for block in self.completed_blocks
            ],
            partial_block=(
                None
                if self.partial_block is None
                else copy_tensor(self.partial_block)
            ),
            outputs_in_current_block=self.outputs_in_current_block,
            total_sublayer_outputs=self.total_sublayer_outputs,
            current_depth_block_index=self.current_depth_block_index,
            block_sizes=list(self.block_sizes),
            inter_block_scan_count=self.inter_block_scan_count,
        )
