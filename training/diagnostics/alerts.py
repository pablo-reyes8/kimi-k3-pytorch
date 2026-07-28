"""Typed, bounded-state diagnostic alert rules."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class DiagnosticAlert:
    severity: Literal["info", "warning", "critical"]
    code: str
    message: str
    step: int
    metric_name: str
    observed_value: float
    reference_value: float | None
    patience_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class AlertManager:
    def __init__(self, patience_steps: int = 5, ema_decay: float = 0.95):
        if patience_steps <= 0 or not 0 <= ema_decay < 1:
            raise ValueError("invalid alert manager configuration")
        self.patience_steps = int(patience_steps)
        self.ema_decay = float(ema_decay)
        self.baselines: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def _condition(
        self,
        metrics,
        name,
        predicate,
        code,
        message,
        step,
        *,
        severity="warning",
        immediate=False,
        reference=None,
    ):
        candidates = {
            key: value
            for key, value in metrics.items()
            if key == name
            or (
                key.startswith(name.split("/", 1)[0] + "/")
                and key.endswith("/" + name.rsplit("/", 1)[-1])
            )
        }
        if not candidates:
            return None
        active_candidates = [
            (key, float(value))
            for key, value in candidates.items()
            if predicate(float(value))
        ]
        active = bool(active_candidates)
        self.counts[code] = self.counts.get(code, 0) + 1 if active else 0
        needed = 1 if immediate else self.patience_steps
        if self.counts[code] < needed:
            return None
        metric_name, value = (
            active_candidates[0]
            if active_candidates
            else next(iter(candidates.items()))
        )
        return DiagnosticAlert(
            severity=severity,
            code=code,
            message=message,
            step=step,
            metric_name=metric_name,
            observed_value=float(value),
            reference_value=reference,
            patience_count=self.counts[code],
        )

    def evaluate(
        self, metrics: dict[str, float], step: int
    ) -> tuple[DiagnosticAlert, ...]:
        alerts = []
        for name, value in metrics.items():
            if not math.isfinite(float(value)):
                alerts.append(
                    DiagnosticAlert(
                        "critical",
                        "NONFINITE_METRIC",
                        f"{name} is non-finite",
                        step,
                        name,
                        float(value),
                        None,
                        1,
                    )
                )
                continue
            previous = self.baselines.get(name, float(value))
            self.baselines[name] = (
                self.ema_decay * previous
                + (1 - self.ema_decay) * float(value)
            )

        rules = [
            (
                "train/loss_total",
                lambda value: not math.isfinite(value),
                "NONFINITE_LOSS",
                "training loss is non-finite",
                "critical",
                True,
                None,
            ),
            (
                "diagnostics/budget_exceeded",
                lambda value: value > 0.5,
                "DIAGNOSTIC_BUDGET_EXCEEDED",
                "diagnostics exceeded the configured time or memory budget",
                "info",
                True,
                0.0,
            ),
            (
                "block/branch_to_input_rms",
                lambda value: value < 1e-3,
                "INACTIVE_BLOCK",
                "a residual branch contribution is persistently negligible",
                "warning",
                False,
                1e-3,
            ),
            (
                "attnres/source_entropy_normalized",
                lambda value: value < 0.05,
                "ATTNRES_SINGLE_SOURCE_COLLAPSE",
                "Attention Residuals collapsed to one depth source",
                "warning",
                False,
                0.05,
            ),
            (
                "attnres/source_entropy_normalized",
                lambda value: value > 0.995,
                "ATTNRES_UNIFORM_COLLAPSE",
                "Attention Residuals remain nearly uniform across sources",
                "info",
                False,
                0.995,
            ),
            (
                "kda/fraction_alpha_near_one",
                lambda value: value > 0.95,
                "KDA_RETENTION_SATURATION",
                "KDA decay is persistently near full retention",
                "warning",
                False,
                0.95,
            ),
            (
                "kda/fraction_beta_near_zero",
                lambda value: value > 0.95,
                "KDA_STATE_WRITE_CLOSED",
                "KDA state writes are persistently closed",
                "warning",
                False,
                0.95,
            ),
            (
                "kda/state_growth_ratio",
                lambda value: value > 2.0,
                "KDA_STATE_EXPLOSION",
                "KDA recurrent state RMS is growing abruptly",
                "warning",
                False,
                2.0,
            ),
            (
                "moe/dead_expert_fraction_batch",
                lambda value: value > 0.5,
                "MOE_DEAD_EXPERTS_PERSISTENT",
                "more than half of routed experts are unused",
                "warning",
                False,
                0.5,
            ),
            (
                "moe/routed_to_total_ratio",
                lambda value: value < 1e-3,
                "MOE_ROUTED_BRANCH_INACTIVE",
                "routed expert branch is negligible",
                "warning",
                False,
                1e-3,
            ),
            (
                "kda/output_gate_saturation_low",
                lambda value: value > 0.95,
                "KDA_OUTPUT_GATE_CLOSED",
                "KDA output gate is persistently closed",
                "warning",
                False,
                0.95,
            ),
            (
                "mla/output_gate_saturation_low",
                lambda value: value > 0.95,
                "MLA_OUTPUT_GATE_CLOSED",
                "MLA output gate is persistently closed",
                "warning",
                False,
                0.95,
            ),
            (
                "qk_clip/fraction_layers_clipped",
                lambda value: value > 0.5,
                "QK_CLIP_PERSISTENT",
                "QK-Clip is active in most attention layers",
                "warning",
                False,
                0.5,
            ),
            (
                "optimizer/mtp/grad_rms",
                lambda value: value <= 1e-12,
                "MTP_DISCONNECTED",
                "MTP parameters receive no measurable sampled gradient",
                "warning",
                False,
                1e-12,
            ),
            (
                "mtp/valid_tokens",
                lambda value: value <= 0,
                "MTP_NO_VALID_TOKENS",
                "MTP repeatedly receives no valid target tokens",
                "warning",
                False,
                0.0,
            ),
            (
                "representation/dead_feature_fraction",
                lambda value: value > 0.9,
                "REPRESENTATION_VARIANCE_COLLAPSE",
                "most sampled representation features have zero variance",
                "warning",
                False,
                0.9,
            ),
        ]
        for name, predicate, code, message, severity, immediate, reference in rules:
            alert = self._condition(
                metrics,
                name,
                predicate,
                code,
                message,
                step,
                severity=severity,
                immediate=immediate,
                reference=reference,
            )
            if alert is not None:
                alerts.append(alert)
        return tuple(alerts)

    def state_dict(self) -> dict:
        return {
            "patience_steps": self.patience_steps,
            "ema_decay": self.ema_decay,
            "baselines": dict(self.baselines),
            "counts": dict(self.counts),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["patience_steps"]) != self.patience_steps:
            raise ValueError("alert patience is incompatible")
        self.baselines = {
            name: float(value)
            for name, value in state.get("baselines", {}).items()
        }
        self.counts = {
            name: int(value)
            for name, value in state.get("counts", {}).items()
        }
