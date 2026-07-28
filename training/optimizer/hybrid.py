"""Transactional Kimi Muon + AdamW optimizer wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import KimiOptimizerConfig
from .muon import KimiMuon
from .parameter_registry import (
    ParameterAssignmentReport,
    build_parameter_registry,
)
from .qk_clip import QKClipController


@dataclass
class OptimizerStepReport:
    executed: bool
    adamw_lr: float
    muon_lr: float
    qk_clip_applied: bool
    qk_clip_events: int
    nonfinite_parameters: int
    update_metrics: dict[str, float]


class KimiHybridOptimizer:
    def __init__(
        self,
        muon: KimiMuon,
        adamw: torch.optim.AdamW,
        report: ParameterAssignmentReport,
        config: KimiOptimizerConfig,
        qk_clip: QKClipController | None,
    ):
        self.muon = muon
        self.adamw = adamw
        self.report = report
        self.config = config
        self.qk_clip = qk_clip
        self.step_number = 0
        self.last_step_report: OptimizerStepReport | None = None

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    @property
    def defaults(self):
        return {"muon": self.muon.defaults, "adamw": self.adamw.defaults}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None) -> OptimizerStepReport:
        if closure is not None:
            with torch.enable_grad():
                closure()
        parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        nonfinite = sum(
            not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        if nonfinite:
            report = OptimizerStepReport(
                executed=False,
                adamw_lr=self.adamw.param_groups[0]["lr"],
                muon_lr=self.muon.param_groups[0]["lr"],
                qk_clip_applied=False,
                qk_clip_events=0,
                nonfinite_parameters=nonfinite,
                update_metrics={},
            )
            self.last_step_report = report
            return report

        update_metrics = self.muon.step()
        self.adamw.step()
        self.step_number += 1
        clip_report = (
            self.qk_clip.apply(self.step_number)
            if self.qk_clip is not None
            else None
        )
        if clip_report is not None:
            update_metrics.update(clip_report.metrics())
        report = OptimizerStepReport(
            executed=True,
            adamw_lr=self.adamw.param_groups[0]["lr"],
            muon_lr=self.muon.param_groups[0]["lr"],
            qk_clip_applied=bool(
                clip_report is not None and clip_report.applied
            ),
            qk_clip_events=0 if clip_report is None else clip_report.events,
            nonfinite_parameters=0,
            update_metrics=update_metrics,
        )
        self.last_step_report = report
        return report

    def set_lr(self, lr: float, muon_lr: float | None = None) -> None:
        if lr < 0 or (muon_lr is not None and muon_lr < 0):
            raise ValueError("learning rates must be non-negative")
        for group in self.adamw.param_groups:
            group["lr"] = lr
        for group in self.muon.param_groups:
            group["lr"] = lr if muon_lr is None else muon_lr

    def set_lrs(self, *, adamw_lr: float, muon_lr: float) -> None:
        self.set_lr(adamw_lr, muon_lr)

    def get_lrs(self) -> dict[str, float]:
        return {
            "adamw_lr": float(self.adamw.param_groups[0]["lr"]),
            "muon_lr": float(self.muon.param_groups[0]["lr"]),
        }

    def parameter_assignment_report(self) -> ParameterAssignmentReport:
        return self.report

    def state_dict(self) -> dict:
        return {
            "format_version": 1,
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
            "parameter_registry_fingerprint": self.report.fingerprint,
            "parameter_registry": self.report.to_dict(),
            "config": self.config.to_dict(),
            "qk_clip": (
                None if self.qk_clip is None else self.qk_clip.state_dict()
            ),
            "step_number": self.step_number,
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        loaded_fingerprint = state.get("parameter_registry_fingerprint")
        if strict and loaded_fingerprint != self.report.fingerprint:
            raise ValueError("optimizer parameter registry fingerprint mismatch")
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        if self.qk_clip is not None and state.get("qk_clip") is not None:
            self.qk_clip.load_state_dict(state["qk_clip"])
        self.step_number = int(state.get("step_number", 0))


def build_kimi_optimizer(
    model,
    config: KimiOptimizerConfig | None = None,
) -> tuple[torch.optim.Optimizer | KimiHybridOptimizer, object]:
    config = config or KimiOptimizerConfig()
    registry = build_parameter_registry(
        model,
        kind=config.kind,
        strict=config.fail_on_unclassified_matrix,
    )
    if config.kind == "adamw":
        groups = [
            {
                "params": [
                    spec.parameter
                    for spec in registry.specs
                    if spec.optimizer_family == "adamw_decay"
                ],
                "weight_decay": config.weight_decay,
                "group_name": "adamw_decay",
            },
            {
                "params": [
                    spec.parameter
                    for spec in registry.specs
                    if spec.optimizer_family == "adamw_no_decay"
                ],
                "weight_decay": 0.0,
                "group_name": "adamw_no_decay",
            },
        ]
        return (
            torch.optim.AdamW(
                groups,
                lr=config.adamw_lr,
                betas=config.adamw_betas,
                eps=config.adamw_eps,
            ),
            registry,
        )

    muon_specs = [
        spec
        for spec in registry.specs
        if spec.optimizer_family in {"muon", "per_head_muon"}
    ]
    if not muon_specs:
        raise RuntimeError("Kimi optimizer found no Muon matrices")
    muon = KimiMuon(
        [{"params": [spec.parameter for spec in muon_specs]}],
        lr=config.resolved_muon_lr,
        momentum=config.muon_momentum,
        nesterov=config.muon_nesterov,
        ns_steps=config.muon_ns_steps,
        eps=config.muon_ns_eps,
        weight_decay=config.resolved_muon_weight_decay,
        update_rms_scaling=config.muon_update_rms_scaling,
        spec_by_parameter={id(spec.parameter): spec for spec in muon_specs},
    )
    decay = [
        spec.parameter
        for spec in registry.specs
        if spec.optimizer_family == "adamw_decay"
    ]
    no_decay = [
        spec.parameter
        for spec in registry.specs
        if spec.optimizer_family == "adamw_no_decay"
    ]
    adamw = torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": config.weight_decay,
                "group_name": "adamw_decay",
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "group_name": "adamw_no_decay",
            },
        ],
        lr=config.adamw_lr,
        betas=config.adamw_betas,
        eps=config.adamw_eps,
    )
    qk_clip = (
        QKClipController(
            model,
            threshold=config.qk_clip_threshold,
            eps=config.qk_clip_eps,
            every_steps=config.qk_clip_every_steps,
            include_kda_experimental=config.qk_clip_kda_experimental,
        )
        if config.qk_clip_enabled
        else None
    )
    optimizer = KimiHybridOptimizer(
        muon, adamw, registry, config, qk_clip
    )
    return optimizer, registry
