"""Auditable tensor-parallel primitives used by the Kimi sharding plan."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .collectives import group_rank, group_size


def _partition(total: int, group, name: str) -> tuple[int, int]:
    size = group_size(group)
    if total % size:
        raise ValueError(f"{name}={total} must be divisible by TP size={size}")
    width = total // size
    start = group_rank(group) * width
    return start, start + width


class _CopyToRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, gradient):
        if group_size(ctx.group) > 1:
            dist.all_reduce(gradient, group=ctx.group)
        return gradient, None


class _ReduceFromRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        output = tensor.clone()
        if group_size(group) > 1:
            dist.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class _GatherLastDim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        ctx.local_width = tensor.shape[-1]
        if group_size(group) == 1:
            return tensor
        pieces = [torch.empty_like(tensor) for _ in range(group_size(group))]
        dist.all_gather(pieces, tensor.contiguous(), group=group)
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def backward(ctx, gradient):
        if group_size(ctx.group) == 1:
            return gradient, None
        start = group_rank(ctx.group) * ctx.local_width
        return gradient.narrow(-1, start, ctx.local_width).contiguous(), None


def copy_to_tensor_parallel_region(tensor: torch.Tensor, group=None):
    return _CopyToRegion.apply(tensor, group)


def reduce_from_tensor_parallel_region(tensor: torch.Tensor, group=None):
    return _ReduceFromRegion.apply(tensor, group)


def gather_from_tensor_parallel_region(tensor: torch.Tensor, group=None):
    return _GatherLastDim.apply(tensor, group)


class ColumnParallelLinear(nn.Module):
    """Shard a linear layer's output rows across TP ranks."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        group=None,
        gather_output: bool = False,
    ):
        super().__init__()
        start, end = _partition(
            linear.out_features, group, "linear.out_features"
        )
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.local_out_features = end - start
        self.output_start = start
        self.output_end = end
        self.group = group
        self.gather_output = gather_output
        self._kimi_parallel_linear = True
        self._kimi_role = "dense_matrix"
        self._kimi_head_spec = None
        self.weight = nn.Parameter(
            linear.weight.detach()[start:end].clone(),
            requires_grad=linear.weight.requires_grad,
        )
        self.bias = (
            None
            if linear.bias is None
            else nn.Parameter(
                linear.bias.detach()[start:end].clone(),
                requires_grad=linear.bias.requires_grad,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        replicated = copy_to_tensor_parallel_region(inputs, self.group)
        local = F.linear(replicated, self.weight, self.bias)
        return (
            gather_from_tensor_parallel_region(local, self.group)
            if self.gather_output
            else local
        )


class RowParallelLinear(nn.Module):
    """Shard a linear layer's input columns and sum partial outputs."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        group=None,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        start, end = _partition(
            linear.in_features, group, "linear.in_features"
        )
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.local_in_features = end - start
        self.input_start = start
        self.input_end = end
        self.group = group
        self.input_is_parallel = input_is_parallel
        self._kimi_parallel_linear = True
        self._kimi_role = "dense_matrix"
        self._kimi_head_spec = None
        self.weight = nn.Parameter(
            linear.weight.detach()[:, start:end].clone(),
            requires_grad=linear.weight.requires_grad,
        )
        self.bias = (
            None
            if linear.bias is None
            else nn.Parameter(
                linear.bias.detach().clone(),
                requires_grad=linear.bias.requires_grad,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local_input = (
            inputs
            if self.input_is_parallel
            else inputs[..., self.input_start : self.input_end]
        )
        partial = F.linear(local_input, self.weight, None)
        output = reduce_from_tensor_parallel_region(partial, self.group)
        if self.bias is not None:
            output = output + self.bias
        return output


class VocabParallelEmbedding(nn.Module):
    """Contiguous vocabulary-sharded embedding with replicated outputs."""

    def __init__(self, embedding: nn.Embedding, *, group=None):
        super().__init__()
        start, end = _partition(
            embedding.num_embeddings, group, "vocab_size"
        )
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.vocab_start = start
        self.vocab_end = end
        self.group = group
        self.padding_idx = embedding.padding_idx
        self._kimi_role = "embedding"
        self.weight = nn.Parameter(
            embedding.weight.detach()[start:end].clone(),
            requires_grad=embedding.weight.requires_grad,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outside = (input_ids < self.vocab_start) | (
            input_ids >= self.vocab_end
        )
        local_ids = (input_ids - self.vocab_start).masked_fill(outside, 0)
        local_padding = (
            self.padding_idx - self.vocab_start
            if self.padding_idx is not None
            and self.vocab_start <= self.padding_idx < self.vocab_end
            else None
        )
        output = F.embedding(
            local_ids,
            self.weight,
            padding_idx=local_padding,
        )
        output = output.masked_fill(outside[..., None], 0)
        return reduce_from_tensor_parallel_region(output, self.group)


class VocabParallelLMHead(nn.Module):
    """Vocabulary-sharded head with optional API-compatible gathered logits."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        group=None,
        gather_output: bool = True,
        tied_weight: nn.Parameter | None = None,
    ):
        super().__init__()
        start, end = _partition(linear.out_features, group, "vocab_size")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.local_out_features = end - start
        self.vocab_start = start
        self.vocab_end = end
        self.group = group
        self.gather_output = gather_output
        self._kimi_role = "lm_head"
        self.weight = (
            tied_weight
            if tied_weight is not None
            else nn.Parameter(
                linear.weight.detach()[start:end].clone(),
                requires_grad=linear.weight.requires_grad,
            )
        )
        self.bias = (
            None
            if linear.bias is None
            else nn.Parameter(
                linear.bias.detach()[start:end].clone(),
                requires_grad=linear.bias.requires_grad,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local = F.linear(
            copy_to_tensor_parallel_region(hidden_states, self.group),
            self.weight,
            self.bias,
        )
        return (
            gather_from_tensor_parallel_region(local, self.group)
            if self.gather_output
            else local
        )


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_logits, targets, start, end, ignore_index, group):
        float_logits = local_logits.float()
        local_max = float_logits.max(dim=-1).values
        global_max = local_max.clone()
        if group_size(group) > 1:
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
        exponentials = torch.exp(float_logits - global_max[..., None])
        denominator = exponentials.sum(dim=-1)
        if group_size(group) > 1:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=group)
        probabilities = exponentials / denominator[..., None]
        local_target = (targets >= start) & (targets < end)
        safe_target = (targets - start).masked_fill(~local_target, 0)
        selected = float_logits.gather(
            -1, safe_target[..., None]
        ).squeeze(-1)
        selected = selected.masked_fill(~local_target, 0)
        if group_size(group) > 1:
            dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=group)
        valid = targets != ignore_index
        losses = (global_max + denominator.log() - selected).masked_fill(
            ~valid, 0
        )
        ctx.save_for_backward(probabilities, safe_target, local_target, valid)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        probabilities, safe_target, local_target, valid = ctx.saved_tensors
        gradient = probabilities
        subtraction = torch.zeros_like(gradient)
        subtraction.scatter_(-1, safe_target[..., None], 1.0)
        gradient = gradient - subtraction * local_target[..., None]
        gradient = gradient * valid[..., None] * grad_output[..., None]
        return gradient, None, None, None, None, None


def vocab_parallel_cross_entropy(
    local_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    vocab_start: int,
    vocab_end: int,
    ignore_index: int = -100,
    group=None,
) -> torch.Tensor:
    """Return unreduced CE while never materializing global logits."""
    if local_logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match logits except for vocabulary axis")
    return _VocabParallelCrossEntropy.apply(
        local_logits,
        targets,
        vocab_start,
        vocab_end,
        ignore_index,
        group,
    )


@dataclass(frozen=True)
class TensorParallelMetadata:
    size: int
    rank: int
    vocab_start: int | None
    vocab_end: int | None
    cache_layout: str
    transformed_attention_layers: int = 0


def shard_vocabulary(
    model: nn.Module, *, group=None, gather_logits: bool = True
) -> TensorParallelMetadata:
    """Replace Kimi's tied embedding/head with one shared local shard."""
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    sharded_embedding = VocabParallelEmbedding(embedding, group=group)
    sharded_head = VocabParallelLMHead(
        head,
        group=group,
        gather_output=gather_logits,
        tied_weight=(
            sharded_embedding.weight
            if head.weight is embedding.weight
            else None
        ),
    )
    model.embed_tokens = sharded_embedding
    model.lm_head = sharded_head
    if getattr(model, "mtp", None) is not None:
        model.mtp.set_shared_modules(sharded_embedding, sharded_head)
    return TensorParallelMetadata(
        size=group_size(group),
        rank=group_rank(group),
        vocab_start=sharded_embedding.vocab_start,
        vocab_end=sharded_embedding.vocab_end,
        cache_layout="replicated",
    )


__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TensorParallelMetadata",
    "VocabParallelEmbedding",
    "VocabParallelLMHead",
    "copy_to_tensor_parallel_region",
    "gather_from_tensor_parallel_region",
    "reduce_from_tensor_parallel_region",
    "shard_vocabulary",
    "vocab_parallel_cross_entropy",
]
