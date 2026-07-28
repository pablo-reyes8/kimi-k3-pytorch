# ============================================================
# EMA UTILITIES
# Generic training stack
# ============================================================

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional, Union, Iterable

import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class EMA:
    """
    Exponential Moving Average over trainable model parameters.

    Exponential moving average for any PyTorch model.

    Features:
        - Shadow weights stored in fp32.
        - Optional CPU offload.
        - Name-based parameter tracking.
        - Safe temporary swap for EMA evaluation.
        - Compatible with checkpoint state_dict.
        - Supports update_after_step and update_every.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[Union[str, torch.device]] = None,
        use_num_updates: bool = True,
        update_after_step: int = 0,
        update_every: int = 1,
        exclude_names: Optional[Iterable[str]] = None,
    ):
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"EMA decay must satisfy 0 <= decay < 1. Got {decay}.")

        if update_after_step < 0:
            raise ValueError(f"update_after_step must be >= 0. Got {update_after_step}.")

        if update_every <= 0:
            raise ValueError(f"update_every must be > 0. Got {update_every}.")

        self.decay = float(decay)
        self.device = torch.device(device) if device is not None else None
        self.use_num_updates = bool(use_num_updates)
        self.update_after_step = int(update_after_step)
        self.update_every = int(update_every)

        self.num_updates = 0
        self.total_steps_seen = 0

        self.exclude_names = tuple(exclude_names or ())

        model = unwrap_model(model)

        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            if self._is_excluded(name):
                continue

            shadow = p.detach().to(dtype=torch.float32).clone()

            if self.device is not None:
                shadow = shadow.to(self.device)

            self.shadow[name] = shadow

    def _is_excluded(self, name: str) -> bool:
        return any(pattern in name for pattern in self.exclude_names)

    def _compute_decay(self) -> float:
        """
        Compute effective decay.

        Warmup formula:
            starts lower and approaches target decay.
        """
        if not self.use_num_updates:
            return self.decay

        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        return float(d)

    @torch.no_grad()
    def update(self, model: nn.Module, step: Optional[int] = None) -> bool:
        """
        Update EMA shadow from current model parameters.

        Args:
            model:
                Live model.
            step:
                Optional global step. If not provided, internal counter is used.

        Returns:
            True if EMA was updated, False otherwise.
        """
        if step is None:
            self.total_steps_seen += 1
            current_step = self.total_steps_seen
        else:
            current_step = int(step)
            self.total_steps_seen = max(self.total_steps_seen, current_step)

        if current_step < self.update_after_step:
            return False

        if (current_step - self.update_after_step) % self.update_every != 0:
            return False

        model = unwrap_model(model)

        self.num_updates += 1
        decay = self._compute_decay()

        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue

            if not p.requires_grad:
                continue

            shadow = self.shadow[name]
            param_fp32 = p.detach().to(dtype=torch.float32)

            if shadow.device != param_fp32.device:
                param_fp32 = param_fp32.to(shadow.device)

            shadow.mul_(decay).add_(param_fp32, alpha=1.0 - decay)

        return True

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """
        Copy EMA weights into the live model.
        """
        model = unwrap_model(model)

        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(device=p.device, dtype=p.dtype))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """
        Store current live model parameters before EMA evaluation.
        """
        model = unwrap_model(model)

        self.backup = {}

        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """
        Restore live model parameters after EMA evaluation.
        """
        model = unwrap_model(model)

        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].to(device=p.device, dtype=p.dtype))

        self.backup = {}

    @contextmanager
    def average_parameters(self, model: nn.Module):
        """
        Temporarily evaluate with EMA weights.

        Usage:
            with ema.average_parameters(model):
                val_loss = evaluate(...)
        """
        self.store(model)
        self.copy_to(model)

        try:
            yield
        finally:
            self.restore(model)

    @torch.no_grad()
    def to(self, device: Union[str, torch.device]) -> None:
        """
        Move EMA shadow weights to a device.
        """
        self.device = torch.device(device)

        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(self.device)

    @torch.no_grad()
    def reinit_from_model(self, model: nn.Module) -> None:
        """
        Hard reset EMA weights from current model weights.
        """
        model = unwrap_model(model)

        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].data.copy_(
                    p.detach().to(device=self.shadow[name].device, dtype=torch.float32)
                )

    def state_dict(self) -> Dict:
        """
        Checkpoint-safe EMA state.
        """
        return {
            "decay": self.decay,
            "device": str(self.device) if self.device is not None else None,
            "use_num_updates": self.use_num_updates,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
            "total_steps_seen": self.total_steps_seen,
            "exclude_names": self.exclude_names,
            "shadow": {
                name: tensor.detach().cpu()
                for name, tensor in self.shadow.items()
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: Dict, strict: bool = False) -> None:
        """
        Restore EMA state from checkpoint.
        """
        self.decay = float(state_dict.get("decay", self.decay))
        self.use_num_updates = bool(state_dict.get("use_num_updates", self.use_num_updates))
        self.update_after_step = int(state_dict.get("update_after_step", self.update_after_step))
        self.update_every = int(state_dict.get("update_every", self.update_every))
        self.num_updates = int(state_dict.get("num_updates", self.num_updates))
        self.total_steps_seen = int(state_dict.get("total_steps_seen", self.total_steps_seen))
        self.exclude_names = tuple(state_dict.get("exclude_names", self.exclude_names))

        loaded_shadow = state_dict.get("shadow", {})

        missing = []
        unexpected = []

        for name, shadow in self.shadow.items():
            if name in loaded_shadow:
                shadow.data.copy_(
                    loaded_shadow[name].to(device=shadow.device, dtype=shadow.dtype)
                )
            else:
                missing.append(name)

        for name in loaded_shadow:
            if name not in self.shadow:
                unexpected.append(name)

        if strict and (missing or unexpected):
            raise RuntimeError(
                f"EMA state mismatch. Missing={missing[:10]}, Unexpected={unexpected[:10]}"
            )

        if missing:
            print(f"[EMA] Warning: {len(missing)} missing EMA parameters.")

        if unexpected:
            print(f"[EMA] Warning: {len(unexpected)} unexpected EMA parameters.")

    def __len__(self) -> int:
        return len(self.shadow)


@torch.no_grad()
def ema_health(
    ema: EMA,
    model: nn.Module,
    rel_tol: float = 5.0,
):
    """
    Basic sanity check comparing EMA weights against live model weights.

    Returns:
        ok, status, relative_difference
    """
    model = unwrap_model(model)

    model_params = []
    ema_params = []

    for name, p in model.named_parameters():
        if name in ema.shadow:
            model_params.append(p.detach().float().cpu().reshape(-1))
            ema_params.append(ema.shadow[name].detach().float().cpu().reshape(-1))

    if not model_params:
        return False, "empty_ema", float("inf")

    model_flat = torch.cat(model_params, dim=0)
    ema_flat = torch.cat(ema_params, dim=0)

    if not torch.isfinite(ema_flat).all():
        return False, "nan_or_inf_in_ema", float("inf")

    model_norm = model_flat.norm().item()
    ema_norm = ema_flat.norm().item()

    if model_norm < 1e-12:
        return False, "model_zero_norm", float("inf")

    if ema_norm < 1e-12:
        return False, "ema_zero_norm", float("inf")

    rel_diff = (model_flat - ema_flat).norm().item() / (model_norm + 1e-8)

    if rel_diff > rel_tol:
        return False, "large_rel_diff", rel_diff

    return True, "ok", rel_diff
