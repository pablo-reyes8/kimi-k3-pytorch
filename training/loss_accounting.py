"""Loss extraction and token-exact accumulation for model outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


def get_output_value(output: Any, name: str, default=None):
    if isinstance(output, dict):
        return output.get(name, default)
    return getattr(output, name, default)


def count_batch_tokens(batch: dict[str, Any], ignore_index: int = -100) -> int:
    input_ids = batch.get("input_ids")
    if not torch.is_tensor(input_ids):
        return 0
    valid = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask = batch.get("attention_mask")
    if torch.is_tensor(attention_mask):
        valid &= attention_mask.bool()
    labels = batch.get("labels")
    if torch.is_tensor(labels):
        valid &= labels.ne(ignore_index)
    loss_mask = batch.get("loss_mask")
    if torch.is_tensor(loss_mask):
        valid &= loss_mask.bool()
    return int(valid.sum().item())


@dataclass
class LossContribution:
    """One microbatch's differentiable sums and detached counters."""

    ntp_loss_sum: torch.Tensor
    ntp_normalizer: float
    mtp_loss_sum: torch.Tensor | None
    mtp_normalizer: float
    lambda_mtp: float
    reported_loss: float
    batch_tokens: int

    @property
    def ntp_loss(self) -> float:
        return float(self.ntp_loss_sum.detach().item()) / max(
            self.ntp_normalizer, 1.0
        )

    @property
    def mtp_loss(self) -> float | None:
        if self.mtp_loss_sum is None:
            return None
        return float(self.mtp_loss_sum.detach().item()) / max(
            self.mtp_normalizer, 1.0
        )


def extract_loss_contribution(
    output: Any,
    batch: dict[str, Any],
) -> LossContribution:
    """Prefer Kimi's typed loss sums; use token-weighted scalar fallback."""

    loss = get_output_value(output, "loss")
    if loss is None or not torch.is_tensor(loss) or loss.ndim != 0:
        raise ValueError("model output must contain a loss scalar")

    loss_output = get_output_value(output, "loss_output")
    ntp = get_output_value(loss_output, "ntp") if loss_output is not None else None
    if ntp is not None:
        ntp_sum = get_output_value(ntp, "loss_sum")
        ntp_normalizer = float(
            get_output_value(ntp, "normalizer").detach().item()
        )
        mtp = get_output_value(loss_output, "mtp")
        mtp_sum = None if mtp is None else get_output_value(mtp, "loss_sum")
        mtp_normalizer = (
            0.0
            if mtp is None
            else float(get_output_value(mtp, "normalizer").detach().item())
        )
        return LossContribution(
            ntp_loss_sum=ntp_sum,
            ntp_normalizer=ntp_normalizer,
            mtp_loss_sum=mtp_sum,
            mtp_normalizer=mtp_normalizer,
            lambda_mtp=float(get_output_value(loss_output, "lambda_mtp", 0.0)),
            reported_loss=float(loss.detach().item()),
            batch_tokens=count_batch_tokens(batch),
        )

    normalizer = float(max(count_batch_tokens(batch), 1))
    return LossContribution(
        ntp_loss_sum=loss * normalizer,
        ntp_normalizer=normalizer,
        mtp_loss_sum=None,
        mtp_normalizer=0.0,
        lambda_mtp=0.0,
        reported_loss=float(loss.detach().item()),
        batch_tokens=count_batch_tokens(batch),
    )


def combine_window_loss(
    contributions: list[LossContribution],
) -> tuple[torch.Tensor, dict[str, float]]:
    if not contributions:
        raise ValueError("cannot combine an empty accumulation window")
    ntp_sum = torch.stack([item.ntp_loss_sum for item in contributions]).sum()
    ntp_normalizer = sum(item.ntp_normalizer for item in contributions)
    if ntp_normalizer <= 0:
        raise ValueError("accumulation window contains no valid NTP tokens")
    objective = ntp_sum / ntp_normalizer

    mtp_items = [item for item in contributions if item.mtp_loss_sum is not None]
    mtp_loss = None
    mtp_normalizer = sum(item.mtp_normalizer for item in mtp_items)
    if mtp_items:
        lambda_values = {item.lambda_mtp for item in mtp_items}
        if len(lambda_values) != 1:
            raise ValueError("lambda_mtp changed inside an accumulation window")
        mtp_sum = torch.stack([item.mtp_loss_sum for item in mtp_items]).sum()
        if mtp_normalizer <= 0:
            raise ValueError("MTP output contains no valid tokens")
        mtp_loss = mtp_sum / mtp_normalizer
        objective = objective + lambda_values.pop() * mtp_loss

    return objective, {
        "loss": float(objective.detach().item()),
        "ntp_loss": float((ntp_sum / ntp_normalizer).detach().item()),
        "mtp_loss": (
            float("nan")
            if mtp_loss is None
            else float(mtp_loss.detach().item())
        ),
        "ntp_tokens": float(ntp_normalizer),
        "mtp_tokens": float(mtp_normalizer),
        "tokens": float(sum(item.batch_tokens for item in contributions)),
    }


def combine_distributed_window_loss(
    contributions: list[LossContribution],
    context,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-exact DP/EP objective accounting with TP replicas excluded."""
    if context is None or not context.initialized:
        return combine_window_loss(contributions)
    if not contributions:
        raise ValueError("cannot combine an empty accumulation window")
    ntp_sum = torch.stack([item.ntp_loss_sum for item in contributions]).sum()
    mtp_items = [item for item in contributions if item.mtp_loss_sum is not None]
    local = torch.tensor(
        [
            sum(item.ntp_normalizer for item in contributions),
            sum(item.mtp_normalizer for item in mtp_items),
            sum(item.batch_tokens for item in contributions),
            float(ntp_sum.detach()),
            (
                0.0
                if not mtp_items
                else float(
                    torch.stack(
                        [item.mtp_loss_sum for item in mtp_items]
                    ).sum().detach()
                )
            ),
        ],
        device=ntp_sum.device,
        dtype=torch.float64,
    )
    for group, size in (
        (context.dp_group, context.dp_size),
        (context.ep_group, context.ep_size),
    ):
        if size > 1:
            dist.all_reduce(local, group=group)
    replicas = context.dp_size * context.ep_size
    if local[0] <= 0:
        raise ValueError("distributed window contains no valid NTP tokens")
    objective = ntp_sum * replicas / local[0].to(ntp_sum.dtype)
    mtp_loss = None
    if mtp_items:
        lambda_values = {item.lambda_mtp for item in mtp_items}
        if len(lambda_values) != 1:
            raise ValueError("lambda_mtp changed inside an accumulation window")
        mtp_sum = torch.stack(
            [item.mtp_loss_sum for item in mtp_items]
        ).sum()
        if local[1] <= 0:
            raise ValueError("distributed MTP output contains no valid tokens")
        mtp_loss = mtp_sum * replicas / local[1].to(mtp_sum.dtype)
        objective = objective + lambda_values.pop() * mtp_loss
    global_ntp_loss = float((local[3] / local[0]).item())
    global_mtp_loss = (
        float("nan")
        if not mtp_items
        else float((local[4] / local[1]).item())
    )
    lambda_mtp = mtp_items[0].lambda_mtp if mtp_items else 0.0
    return objective, {
        "loss": global_ntp_loss + lambda_mtp * (
            0.0 if not mtp_items else global_mtp_loss
        ),
        "ntp_loss": global_ntp_loss,
        "mtp_loss": global_mtp_loss,
        "ntp_tokens": float(local[0].item()),
        "mtp_tokens": float(local[1].item()),
        "tokens": float(local[2].item()),
    }
