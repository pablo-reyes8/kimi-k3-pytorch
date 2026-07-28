"""Standard and Per-Head Muon in one auditable optimizer backend."""

from __future__ import annotations

import math

import torch

from .newton_schulz import match_update_rms, zeropower_via_newton_schulz
from .per_head_muon import per_head_orthogonalize


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().square().mean().sqrt().item())


class KimiMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float,
        momentum: float,
        nesterov: bool,
        ns_steps: int,
        eps: float,
        weight_decay: float,
        update_rms_scaling: bool,
        spec_by_parameter: dict[int, object],
    ):
        if lr <= 0 or not 0 <= momentum < 1:
            raise ValueError("invalid Muon learning rate or momentum")
        if ns_steps <= 0 or eps <= 0 or weight_decay < 0:
            raise ValueError("invalid Muon numerical configuration")
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                eps=eps,
                weight_decay=weight_decay,
                update_rms_scaling=update_rms_scaling,
            ),
        )
        self.spec_by_parameter = spec_by_parameter
        self.last_update_metrics: dict[str, float] = {}
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError("Muon parameters must be 2D")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        gradients = [
            parameter.grad
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError(
                "non-finite Muon gradient detected before state mutation"
            )

        raw_values = []
        ortho_values = []
        scaled_values = []
        ratios = []
        per_head_metrics: dict[str, list[float]] = {}
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                state = self.state[parameter]
                buffer = state.setdefault(
                    "momentum_buffer", torch.zeros_like(grad)
                )
                buffer.mul_(group["momentum"]).add_(grad)
                raw = (
                    grad.add(buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else buffer
                )
                raw_values.append(_rms(raw))
                spec = self.spec_by_parameter[id(parameter)]
                if spec.optimizer_family == "per_head_muon":
                    update, head_metrics = per_head_orthogonalize(
                        raw,
                        spec.head_layout,
                        steps=group["ns_steps"],
                        eps=group["eps"],
                        rms_scaling=group["update_rms_scaling"],
                    )
                    for key, value in head_metrics.items():
                        per_head_metrics.setdefault(key, []).append(value)
                    ortho_values.append(_rms(update))
                else:
                    update = zeropower_via_newton_schulz(
                        raw,
                        steps=group["ns_steps"],
                        eps=group["eps"],
                    )
                    ortho_values.append(_rms(update))
                    if group["update_rms_scaling"]:
                        update = match_update_rms(update, mode="shape")
                scaled = _rms(update)
                scaled_values.append(scaled)
                parameter_rms = _rms(parameter)
                ratios.append(
                    group["lr"] * scaled / max(parameter_rms, group["eps"])
                )
                if group["weight_decay"]:
                    parameter.mul_(
                        1.0 - group["lr"] * group["weight_decay"]
                    )
                parameter.add_(update, alpha=-group["lr"])

        def mean(values):
            return sum(values) / len(values) if values else 0.0

        raw_mean = mean(raw_values)
        scaled_mean = mean(scaled_values)
        self.last_update_metrics = {
            "muon/raw_update_rms": raw_mean,
            "muon/orthogonal_update_rms": mean(ortho_values),
            "muon/scaled_update_rms": scaled_mean,
            "muon/ns_gain": scaled_mean / max(raw_mean, 1e-12),
            "muon/update_to_parameter_ratio": mean(ratios),
        }
        self.last_update_metrics.update(
            {
                key: mean(values)
                for key, values in per_head_metrics.items()
            }
        )
        return dict(self.last_update_metrics)
