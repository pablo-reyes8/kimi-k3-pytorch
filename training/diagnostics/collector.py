"""Scheduled, zero-graph-retention Kimi diagnostic collector."""

from __future__ import annotations

import time

from .alerts import AlertManager
from .activation_metrics import compute_activation_metrics
from .attnres_metrics import compute_attnres_metrics
from .block_metrics import compute_block_contribution
from .config import DiagnosticsConfig
from .kda_metrics import compute_kda_metrics
from .loss_metrics import compute_loss_metrics
from .mla_metrics import compute_mla_metrics
from .moe_metrics import compute_moe_metrics
from .mtp_metrics import compute_mtp_metrics
from .optimizer_metrics import ParameterUpdateMonitor
from .representation_metrics import compute_representation_metrics
from .reducers import scalar


class KimiDiagnosticCollector:
    def __init__(
        self,
        model,
        config: DiagnosticsConfig | None = None,
        *,
        parameter_specs=(),
    ):
        self.model = model
        self.config = config or DiagnosticsConfig()
        self.alerts = AlertManager(self.config.alert_patience_steps)
        self.update_monitor = ParameterUpdateMonitor(
            parameter_specs,
            max_parameters_per_group=self.config.sample_parameters_per_group,
            max_elements_per_parameter=self.config.sample_elements_per_parameter,
        )
        self.latest_metrics: dict[str, float] = {}
        self.latest_alerts = ()
        self.rotation_index = 0
        self.last_diagnostic_time_ms = 0.0
        self.current_diagnostic_time_ms = 0.0
        self.degrade_until_step = 0
        self.metric_ema: dict[str, float] = {}
        self.previous_state_rms: dict[str, float] = {}
        self.active_level: str | None = None

    def level_for_step(self, step: int) -> str | None:
        level = self.config.level_for_step(step)
        if step <= self.degrade_until_step and level in {"standard", "deep"}:
            return "cheap"
        return level

    def model_kwargs_for_step(self, step: int) -> dict:
        level = self.level_for_step(step)
        if level not in {"standard", "deep"}:
            return {}
        return {
            "output_hidden_states": True,
            "output_router_diagnostics": True,
            "output_attnres_diagnostics": True,
        }

    def _sample_layer_indices(self, count: int, step: int) -> tuple[int, ...]:
        limit = min(self.config.sample_layers_per_standard_step, count)
        if count <= limit:
            return tuple(range(count))
        anchors = tuple(dict.fromkeys((0, count // 2, count - 1)))
        required = set(anchors[:limit])
        cursor = (
            step // self.config.standard_every_steps
            if self.config.rotate_sampled_layers
            else 0
        )
        index = cursor % count
        while len(required) < limit:
            required.add(index)
            index = (index + 1) % count
        return tuple(sorted(required))

    def collect_output(self, output, *, step: int) -> dict[str, float]:
        self.active_level = self.level_for_step(step)
        if self.active_level is None:
            self.latest_metrics = {}
            return {}
        started = time.perf_counter()
        metrics: dict[str, float] = {}
        loss_output = getattr(output, "loss_output", None)
        if loss_output is not None:
            ntp = loss_output.ntp
            mtp = loss_output.mtp
            metrics.update(
                compute_loss_metrics(
                    total_loss=scalar(loss_output.loss),
                    ntp_loss=scalar(ntp.loss),
                    mtp_loss=None if mtp is None else scalar(mtp.loss),
                    ntp_tokens=scalar(ntp.normalizer),
                    mtp_tokens=0 if mtp is None else scalar(mtp.normalizer),
                    lambda_mtp=loss_output.lambda_mtp,
                )
            )

        level = self.active_level
        if level in {"standard", "deep"}:
            backbone = getattr(output, "backbone_diagnostics", None) or {}
            layers = tuple(backbone.get("layers", ()))
            sampled = self._sample_layer_indices(len(layers), step)
            for index in sampled:
                layer = layers[index]
                mechanism = layer.get("mechanism")
                attention_type = layer.get("attention_type", "")
                if attention_type == "kda":
                    metrics.update(
                        compute_kda_metrics(
                            mechanism, prefix=f"kda/layer_{index:02d}"
                        )
                    )
                elif attention_type in {"gated_mla", "gated_mla_final"}:
                    metrics.update(
                        compute_mla_metrics(
                            mechanism, prefix=f"mla/layer_{index:02d}"
                        )
                    )
                metrics.update(
                    compute_moe_metrics(
                        layer.get("channel_mixer"),
                        prefix=f"moe/layer_{index:02d}",
                    )
                )
                dead_name = (
                    f"moe/layer_{index:02d}/dead_expert_fraction_batch"
                )
                if dead_name in metrics:
                    ema_name = (
                        f"moe/layer_{index:02d}/dead_expert_fraction_ema"
                    )
                    previous = self.metric_ema.get(
                        ema_name, metrics[dead_name]
                    )
                    current = 0.95 * previous + 0.05 * metrics[dead_name]
                    self.metric_ema[ema_name] = current
                    metrics[ema_name] = current
                state_name = f"kda/layer_{index:02d}/state_rms"
                if state_name in metrics:
                    growth_name = (
                        f"kda/layer_{index:02d}/state_growth_ratio"
                    )
                    previous = self.previous_state_rms.get(
                        state_name, metrics[state_name]
                    )
                    metrics[growth_name] = metrics[state_name] / max(
                        previous, 1e-12
                    )
                    self.previous_state_rms[state_name] = metrics[state_name]

            trace = getattr(output, "backbone_trace", None)
            if trace is not None:
                for index in sampled:
                    attention_input = trace.pre_attention[index]
                    attention_branch = trace.attention_outputs[index]
                    block_metrics = compute_block_contribution(
                            attention_input,
                            attention_branch,
                            attention_input + attention_branch,
                            prefix=f"block/layer_{index:02d}/attention",
                        )
                    metrics.update(block_metrics)
                    architecture_prefix = (
                        "kda" if layers[index].get("attention_type") == "kda"
                        else "mla"
                    )
                    metrics[
                        f"{architecture_prefix}/layer_{index:02d}/"
                        "branch_to_input_rms"
                    ] = block_metrics[
                        f"block/layer_{index:02d}/attention/"
                        "branch_to_input_rms"
                    ]
                    ffn_input = trace.pre_ffn[index]
                    ffn_branch = trace.ffn_outputs[index]
                    metrics.update(
                        compute_block_contribution(
                            ffn_input,
                            ffn_branch,
                            ffn_input + ffn_branch,
                            prefix=f"block/layer_{index:02d}/moe",
                        )
                    )
                metrics.update(
                    compute_representation_metrics(
                        trace.final_mixed,
                        max_tokens=self.config.sample_tokens_per_layer,
                    )
                )
                metrics.update(
                    compute_activation_metrics(
                        trace.final_mixed,
                        max_elements=self.config.sample_tokens_per_layer
                        * trace.final_mixed.shape[-1],
                    )
                )
            metrics.update(
                compute_attnres_metrics(
                    getattr(output, "attnres_diagnostics", None)
                )
            )
            mtp_diagnostics = getattr(output, "mtp_diagnostics", None)
            metrics.update(
                compute_mtp_metrics(
                    mtp_diagnostics,
                    mtp_loss=metrics.get("train/loss_mtp"),
                    loss_weight=(
                        0.0 if loss_output is None else loss_output.lambda_mtp
                    ),
                )
            )
            metrics["diagnostics/layers_sampled"] = float(len(sampled))

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_diagnostic_time_ms = elapsed_ms
        self.current_diagnostic_time_ms = elapsed_ms
        metrics["diagnostics/time_ms"] = elapsed_ms
        metrics["train/diagnostics_time_ms"] = elapsed_ms
        metrics["diagnostics/scalars_emitted"] = float(len(metrics))
        self.latest_metrics = dict(metrics)
        return dict(metrics)

    def capture_before_optimizer_step(self) -> dict[str, float]:
        if self.active_level is None:
            return {}
        started = time.perf_counter()
        metrics = self.update_monitor.capture_before_step()
        self.current_diagnostic_time_ms += (
            time.perf_counter() - started
        ) * 1000.0
        return metrics

    def capture_after_optimizer_step(
        self,
        *,
        step: int,
        step_time_ms: float | None = None,
    ) -> tuple[dict[str, float], tuple]:
        if self.active_level is None:
            return {}, ()
        started = time.perf_counter()
        metrics = self.update_monitor.capture_after_step()
        self.current_diagnostic_time_ms += (
            time.perf_counter() - started
        ) * 1000.0
        self.last_diagnostic_time_ms = self.current_diagnostic_time_ms
        metrics["diagnostics/time_ms"] = (
            self.last_diagnostic_time_ms
        )
        metrics["train/diagnostics_time_ms"] = (
            self.last_diagnostic_time_ms
        )
        if step_time_ms is not None:
            metrics["diagnostics/time_fraction"] = (
                self.last_diagnostic_time_ms / max(step_time_ms, 1e-12)
            )
            metrics["train/diagnostics_time_fraction"] = metrics[
                "diagnostics/time_fraction"
            ]
        time_exceeded = metrics.get(
            "diagnostics/time_fraction", 0.0
        ) > self.config.max_diagnostic_time_fraction
        memory_exceeded = metrics.get(
            "diagnostics/persistent_gpu_bytes", 0.0
        ) > self.config.max_persistent_gpu_bytes
        budget_exceeded = time_exceeded or memory_exceeded
        if budget_exceeded:
            self.degrade_until_step = max(
                self.degrade_until_step,
                step + self.config.standard_every_steps,
            )
        metrics["diagnostics/budget_exceeded"] = float(budget_exceeded)
        metrics["diagnostics/degradation_level"] = float(
            step <= self.degrade_until_step
        )
        combined = {**self.latest_metrics, **metrics}
        self.latest_alerts = self.alerts.evaluate(combined, step)
        self.latest_metrics = combined
        self.active_level = None
        return dict(metrics), self.latest_alerts

    def state_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "alerts": self.alerts.state_dict(),
            "rotation_index": self.rotation_index,
            "latest_metrics": dict(self.latest_metrics),
            "degrade_until_step": self.degrade_until_step,
            "metric_ema": dict(self.metric_ema),
            "previous_state_rms": dict(self.previous_state_rms),
        }

    def load_state_dict(self, state: dict) -> None:
        self.alerts.load_state_dict(state["alerts"])
        self.rotation_index = int(state.get("rotation_index", 0))
        self.latest_metrics = {
            name: float(value)
            for name, value in state.get("latest_metrics", {}).items()
        }
        self.degrade_until_step = int(state.get("degrade_until_step", 0))
        self.metric_ema = {
            name: float(value)
            for name, value in state.get("metric_ema", {}).items()
        }
        self.previous_state_rms = {
            name: float(value)
            for name, value in state.get("previous_state_rms", {}).items()
        }

    def diagnostic_snapshot(self, step: int) -> dict:
        return {
            "step": int(step),
            "metrics": dict(self.latest_metrics),
            "alerts": [alert.to_dict() for alert in self.latest_alerts],
        }
