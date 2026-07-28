"""Bounded sampled gradient, parameter and update measurements."""

from __future__ import annotations

from collections import defaultdict

import torch


def diagnostic_family(spec) -> str:
    name = spec.parameter_name.lower()
    role = spec.role
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp"
    if role == "embedding":
        return "embeddings"
    if role in {"attention_q", "attention_k", "attention_v"}:
        return "attention_qkv"
    if role == "attention_o":
        return "attention_output"
    if role == "moe_router":
        return "moe_router"
    if role == "moe_shared_expert":
        return "moe_shared_experts"
    if role == "moe_routed_expert":
        return "moe_routed_experts"
    if role == "mtp_matrix":
        return "mtp"
    if role == "lm_head":
        return "lm_head"
    if role == "norm":
        return "norms"
    if "attnres" in name or "pseudo_query" in name:
        return "attnres"
    if ".attention." in name and "kda" in name:
        return "kda_internal"
    return "other"


class ParameterUpdateMonitor:
    """Clone only a short contiguous prefix from a bounded tensor sample."""

    def __init__(
        self,
        specs,
        *,
        max_parameters_per_group: int = 16,
        max_elements_per_parameter: int = 256,
    ):
        self.specs = tuple(specs)
        self.max_parameters_per_group = int(max_parameters_per_group)
        self.max_elements_per_parameter = int(max_elements_per_parameter)
        self.before: dict[str, torch.Tensor] = {}
        self.sampled_specs = []
        self.persistent_bytes = 0

    def capture_before_step(self) -> dict[str, float]:
        grouped = defaultdict(list)
        for spec in self.specs:
            grouped[diagnostic_family(spec)].append(spec)
        self.before.clear()
        self.sampled_specs.clear()
        self.persistent_bytes = 0
        grad_sums = defaultdict(float)
        grad_counts = defaultdict(int)
        zero_counts = defaultdict(int)
        nonfinite = defaultdict(int)
        missing = defaultdict(int)
        for family, specs in grouped.items():
            for spec in specs[: self.max_parameters_per_group]:
                parameter = spec.parameter
                sample = parameter.detach().reshape(-1)[
                    : self.max_elements_per_parameter
                ].float()
                clone = sample.clone()
                self.before[spec.parameter_name] = clone
                self.sampled_specs.append(spec)
                self.persistent_bytes += clone.numel() * clone.element_size()
                gradient = parameter.grad
                if gradient is None:
                    missing[family] += 1
                    continue
                grad_sample = gradient.detach().reshape(-1)[
                    : self.max_elements_per_parameter
                ].float()
                nonfinite[family] += int(
                    (~torch.isfinite(grad_sample)).sum().item()
                )
                finite = torch.nan_to_num(grad_sample)
                grad_sums[family] += float(finite.square().sum().item())
                grad_counts[family] += finite.numel()
                zero_counts[family] += int((finite == 0).sum().item())
        metrics = {}
        main_grad_sum = main_grad_count = 0
        for family in grouped:
            count = grad_counts[family]
            metrics[f"optimizer/{family}/grad_rms"] = (
                (grad_sums[family] / count) ** 0.5 if count else 0.0
            )
            metrics[f"optimizer/{family}/zero_grad_fraction"] = (
                zero_counts[family] / count if count else 0.0
            )
            metrics[f"optimizer/{family}/nonfinite_grad_count"] = float(
                nonfinite[family]
            )
            metrics[f"optimizer/{family}/grad_none_tensors"] = float(
                missing[family]
            )
            if family != "mtp":
                main_grad_sum += grad_sums[family]
                main_grad_count += grad_counts[family]
        global_grad_sum = sum(grad_sums.values())
        global_grad_count = sum(grad_counts.values())
        metrics["train/gradient_rms_global_sampled"] = (
            (global_grad_sum / global_grad_count) ** 0.5
            if global_grad_count else 0.0
        )
        if "mtp" in grouped:
            mtp_grad = metrics["optimizer/mtp/grad_rms"]
            main_grad = (
                (main_grad_sum / main_grad_count) ** 0.5
                if main_grad_count else 0.0
            )
            metrics["mtp/gradient_rms"] = mtp_grad
            metrics["mtp/main_to_mtp_grad_ratio"] = (
                main_grad / max(mtp_grad, 1e-12)
            )
        return metrics

    @torch.no_grad()
    def capture_after_step(self) -> dict[str, float]:
        update_sums = defaultdict(float)
        parameter_sums = defaultdict(float)
        counts = defaultdict(int)
        for spec in self.sampled_specs:
            family = diagnostic_family(spec)
            after = spec.parameter.detach().reshape(-1)[
                : self.max_elements_per_parameter
            ].float()
            before = self.before[spec.parameter_name]
            update = after - before
            update_sums[family] += float(update.square().sum().item())
            parameter_sums[family] += float(after.square().sum().item())
            counts[family] += after.numel()
        metrics = {}
        total_update = total_parameter = total_count = 0
        for family, count in counts.items():
            update_rms = (update_sums[family] / count) ** 0.5
            parameter_rms = (parameter_sums[family] / count) ** 0.5
            metrics[f"optimizer/{family}/parameter_rms"] = parameter_rms
            metrics[f"optimizer/{family}/update_rms"] = update_rms
            metrics[
                f"optimizer/{family}/update_to_parameter_ratio"
            ] = update_rms / max(parameter_rms, 1e-12)
            if family == "mtp":
                metrics["mtp/update_rms"] = update_rms
                metrics["mtp/update_to_parameter_ratio"] = (
                    update_rms / max(parameter_rms, 1e-12)
                )
            total_update += update_sums[family]
            total_parameter += parameter_sums[family]
            total_count += count
        global_update = (total_update / max(total_count, 1)) ** 0.5
        global_parameter = (total_parameter / max(total_count, 1)) ** 0.5
        metrics.update(
            {
                "train/parameter_norm_global_sampled": global_parameter,
                "train/update_norm_global_sampled": global_update,
                "train/update_to_parameter_ratio_sampled": global_update
                / max(global_parameter, 1e-12),
                "diagnostics/persistent_gpu_bytes": float(
                    self.persistent_bytes
                ),
            }
        )
        self.before.clear()
        self.sampled_specs.clear()
        self.persistent_bytes = 0
        return metrics
