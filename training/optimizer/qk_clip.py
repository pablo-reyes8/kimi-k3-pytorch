"""Directed Kimi-style QK weight rescaling."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from src.kda import KimiDeltaAttention
from src.mla import GatedMLA


@dataclass
class QKClipReport:
    applied: bool
    events: int
    fraction_layers_clipped: float
    max_preclip_scale: float
    max_postclip_scale: float
    mean_rescale_factor: float
    min_rescale_factor: float
    consecutive_steps_active: int

    def metrics(self) -> dict[str, float]:
        return {
            "qk_clip/events_step": float(self.events),
            "qk_clip/fraction_layers_clipped": self.fraction_layers_clipped,
            "qk_clip/max_preclip_scale": self.max_preclip_scale,
            "qk_clip/max_postclip_scale": self.max_postclip_scale,
            "qk_clip/mean_rescale_factor": self.mean_rescale_factor,
            "qk_clip/min_rescale_factor": self.min_rescale_factor,
            "qk_clip/consecutive_steps_active": float(
                self.consecutive_steps_active
            ),
        }


class QKClipController:
    """Use the latest detached attention-scale proxy to rescale Q and K."""

    def __init__(
        self,
        model,
        *,
        threshold: float,
        eps: float = 1e-6,
        every_steps: int = 1,
        include_kda_experimental: bool = False,
    ):
        if threshold <= 0 or eps <= 0 or every_steps <= 0:
            raise ValueError("invalid QK-Clip configuration")
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.every_steps = int(every_steps)
        self.layers = []
        for name, module in model.named_modules():
            custom_weights = getattr(module, "_kimi_qk_weights", None)
            if custom_weights is not None:
                self.layers.append(
                    (name, module, custom_weights[0], custom_weights[1])
                )
            elif isinstance(module, GatedMLA):
                self.layers.append(
                    (
                        name,
                        module,
                        module.projections.query.weight,
                        module.projections.latent_kv.key_up.weight,
                    )
                )
            elif include_kda_experimental and isinstance(
                module, KimiDeltaAttention
            ):
                self.layers.append(
                    (
                        name,
                        module,
                        module.projections.q_proj.weight,
                        module.projections.k_proj.weight,
                    )
                )
        self.total_events = 0
        self.consecutive_steps_active = 0

    @torch.no_grad()
    def apply(self, step: int) -> QKClipReport:
        if step % self.every_steps:
            return QKClipReport(
                False, 0, 0.0, 0.0, 0.0, 1.0, 1.0,
                self.consecutive_steps_active,
            )
        observed = []
        factors = []
        for _, module, query_weight, key_weight in self.layers:
            scale = getattr(module, "_last_qk_scale", None)
            if scale is None:
                continue
            reduced_scale = scale.detach().float().clone()
            process_group = getattr(module, "_kimi_qk_group", None)
            if (
                process_group is not None
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size(process_group) > 1
            ):
                torch.distributed.all_reduce(
                    reduced_scale,
                    op=torch.distributed.ReduceOp.MAX,
                    group=process_group,
                )
            value = float(reduced_scale.item())
            if not math.isfinite(value):
                raise FloatingPointError("QK scale proxy is non-finite")
            observed.append(value)
            if value > self.threshold:
                correction = min(
                    1.0, self.threshold / max(value, self.eps)
                )
                symmetric = correction ** 0.5
                query_weight.mul_(symmetric)
                key_weight.mul_(symmetric)
                factors.append(correction)
        events = len(factors)
        self.total_events += events
        self.consecutive_steps_active = (
            self.consecutive_steps_active + 1 if events else 0
        )
        max_pre = max(observed, default=0.0)
        max_post = max(
            [min(value, self.threshold) for value in observed],
            default=0.0,
        )
        return QKClipReport(
            applied=bool(events),
            events=events,
            fraction_layers_clipped=events / max(len(self.layers), 1),
            max_preclip_scale=max_pre,
            max_postclip_scale=max_post,
            mean_rescale_factor=(
                sum(factors) / len(factors) if factors else 1.0
            ),
            min_rescale_factor=min(factors, default=1.0),
            consecutive_steps_active=self.consecutive_steps_active,
        )

    def state_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "eps": self.eps,
            "every_steps": self.every_steps,
            "total_events": self.total_events,
            "consecutive_steps_active": self.consecutive_steps_active,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("threshold", "eps", "every_steps"):
            if float(state[key]) != float(getattr(self, key)):
                raise ValueError(f"QK-Clip {key} is incompatible")
        self.total_events = int(state["total_events"])
        self.consecutive_steps_active = int(
            state["consecutive_steps_active"]
        )
