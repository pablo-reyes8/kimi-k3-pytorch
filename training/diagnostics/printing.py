"""Clear block-based console/Jupyter presentation for Kimi training."""

from __future__ import annotations

import math


def _format(value) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _rule(width=104, character="─") -> str:
    return character * width


class KimiTrainingPrinter:
    def __init__(self, width: int = 104):
        self.width = width

    def title(self, text: str) -> None:
        print("\n" + _rule(self.width, "═"))
        print(text)
        print(_rule(self.width, "═"))

    def block(
        self,
        title: str,
        metrics: dict,
        rows: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    ) -> None:
        available = [(name, label) for name, label in rows if name in metrics]
        if not available:
            return
        print(f"\n  {title}")
        print("  " + _rule(min(self.width - 2, 88)))
        label_width = max(len(label) for _, label in available)
        for name, label in available:
            print(f"  {label:<{label_width}} : {_format(metrics[name]):>12}")

    def print_run_header(
        self,
        *,
        run_name: str,
        model,
        device,
        precision: str,
        optimizer_kind: str,
        epochs: int,
        total_steps: int,
        warmup_steps: int,
        registry=None,
    ) -> None:
        self.title(f"Kimi K3 training · {run_name}")
        print(f"  Device/precision : {device} / {precision}")
        print(f"  Optimizer        : {optimizer_kind}")
        print(
            f"  Schedule         : epochs={epochs} · total_steps={total_steps} "
            f"· warmup_steps={warmup_steps}"
        )
        if registry is not None:
            percentages = registry.percentages_by_family
            print(
                "  Parameter map     : "
                f"AdamW={percentages['adamw_decay'] + percentages['adamw_no_decay']:.1f}% · "
                f"Muon={percentages['muon']:.1f}% · "
                f"Per-Head={percentages['per_head_muon']:.1f}%"
            )
        print(_rule(self.width))

    def print_step(
        self,
        step: int,
        metrics: dict,
        alerts=(),
    ) -> None:
        self.title(f"Optimizer step {step:06d}")
        self.block(
            "Convergence",
            metrics,
            (
                ("train/loss_total", "Total loss"),
                ("train/loss_ntp", "NTP loss"),
                ("train/loss_mtp", "MTP loss"),
                ("train/perplexity_ntp_clipped", "NTP perplexity"),
                ("train/valid_ntp_tokens", "Valid NTP tokens"),
            ),
        )
        self.block(
            "Throughput & runtime",
            metrics,
            (
                ("train/tokens_per_second", "Tokens / second"),
                ("train/samples_per_second", "Samples / second"),
                ("train/step_time_ms", "Step time (ms)"),
                ("train/data_time_ms", "Data time (ms)"),
                ("train/forward_time_ms", "Forward time (ms)"),
                ("train/backward_time_ms", "Backward time (ms)"),
                ("train/optimizer_time_ms", "Optimizer time (ms)"),
                ("train/memory_allocated_mb", "Memory allocated (MB)"),
            ),
        )
        self.block(
            "Optimization health",
            metrics,
            (
                ("learning_rate", "Learning rate"),
                ("lr/adamw", "AdamW LR"),
                ("lr/muon", "Muon LR"),
                ("grad_norm_pre_clip", "Gradient norm pre-clip"),
                ("grad_norm_post_clip", "Gradient norm post-clip"),
                ("train/update_to_parameter_ratio_sampled", "Update / parameter"),
                ("muon/ns_gain", "Muon NS gain"),
                ("per_head_muon/head_update_rms_cv", "Head update RMS CV"),
                ("qk_clip/events_step", "QK-Clip events"),
            ),
        )
        self.block(
            "Diagnostics budget",
            metrics,
            (
                ("train/diagnostics_time_ms", "Diagnostics time (ms)"),
                (
                    "train/diagnostics_time_fraction",
                    "Diagnostics / step",
                ),
                (
                    "diagnostics/persistent_gpu_bytes",
                    "Persistent sampled bytes",
                ),
                ("diagnostics/scalars_emitted", "Scalars emitted"),
                ("diagnostics/layers_sampled", "Layers sampled"),
                ("diagnostics/degradation_level", "Degraded mode"),
            ),
        )
        self.block(
            "Progressive context",
            metrics,
            (
                ("context/stage_index", "Stage"),
                ("context/max_seq_len", "Active max sequence"),
                ("context/tokens_seen", "Tokens seen"),
                ("context/transition_count", "Transitions"),
                ("context/valid_tokens_per_step", "Valid tokens / step"),
                ("context/padding_fraction", "Padding fraction"),
                ("context/tokens_per_second", "Context tokens / second"),
                ("context/step_time_seconds", "Context step time (s)"),
                ("context/peak_memory_mb", "Peak memory (MB)"),
                ("context/old_max_seq_len", "Previous max sequence"),
                ("context/new_max_seq_len", "New max sequence"),
                (
                    "context/tokens_seen_at_transition",
                    "Tokens at transition",
                ),
                (
                    "context/loss_before_transition",
                    "Loss before transition",
                ),
            ),
        )
        architecture_rows = []
        for name in sorted(metrics):
            if any(
                name.startswith(prefix)
                for prefix in (
                    "attnres/", "kda/", "mla/", "moe/", "mtp/", "vision/"
                )
            ):
                if name.endswith(
                    (
                        "source_entropy_normalized",
                        "output_gate_mean",
                        "qk_scale_max",
                        "dead_expert_fraction_batch",
                        "routed_to_total_ratio",
                        "loss",
                        "hidden_rms",
                    )
                ):
                    architecture_rows.append((name, name))
        self.block("Architecture health", metrics, architecture_rows[:16])
        if alerts:
            print("\n  Alerts")
            print("  " + _rule(min(self.width - 2, 88), "·"))
            for alert in alerts:
                print(
                    f"  [{alert.severity.upper():8}] {alert.code}: "
                    f"{alert.message} ({_format(alert.observed_value)})"
                )
        print(_rule(self.width))

    def print_epoch_summary(
        self,
        *,
        epoch: int,
        state,
        train_stats: dict,
        eval_stats: dict | None,
        checkpoint=None,
    ) -> None:
        self.title(f"Epoch {epoch:03d} summary")
        print(
            f"  Optimizer step : {state.optimizer_step} · "
            f"tokens seen: {state.tokens_seen:,}"
        )
        print(
            f"  Train          : loss={_format(train_stats.get('loss'))} · "
            f"NTP={_format(train_stats.get('ntp_loss'))} · "
            f"MTP={_format(train_stats.get('mtp_loss'))}"
        )
        if eval_stats is not None:
            print(
                f"  Evaluation     : loss={_format(eval_stats.get('loss'))} · "
                f"NTP ppl={_format(eval_stats.get('ntp_perplexity'))}"
            )
        if checkpoint is not None:
            print(f"  Checkpoint     : {checkpoint}")
        print(_rule(self.width))
