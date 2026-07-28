# ============================================================
# WARMUP + COSINE SCHEDULER
# Supports AdamW and HybridMuonAdamW
# Generic training stack
# ============================================================

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union


class WarmupCosineLR:
    """
    Step-based linear warmup + cosine decay scheduler.

    Supports:
        1. Standard torch optimizer, e.g. AdamW
        2. HybridMuonAdamW with:
            optimizer.muon
            optimizer.adamw
            optimizer.set_lr(lr, muon_lr=None)

    Behavior:
        - steps 1..warmup_steps:
            lr increases linearly from 0 to base_lr
        - after warmup:
            cosine decay from base_lr to min_lr
        - resume-safe with state_dict/load_state_dict

    Usage:
        scheduler = WarmupCosineLR(
            optimizer=optimizer,
            total_steps=10000,
            warmup_steps=500,
            min_lr=3e-5,
        )

        optimizer.step()
        scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 0.0,
        min_muon_lr: Optional[float] = None,
        prepare_first_update: bool = False,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")

        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        if min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {min_lr}")

        if min_muon_lr is not None and min_muon_lr < 0:
            raise ValueError(f"min_muon_lr must be >= 0, got {min_muon_lr}")

        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.min_muon_lr = float(min_muon_lr) if min_muon_lr is not None else None
        self.prepare_first_update = bool(prepare_first_update)

        self.step_num = 0

        self.is_hybrid = (
            hasattr(optimizer, "muon")
            and hasattr(optimizer, "adamw")
        )

        if self.is_hybrid:
            self.base_adamw_lrs = [
                float(group["lr"])
                for group in optimizer.adamw.param_groups
            ]

            self.base_muon_lrs = [
                float(group["lr"])
                for group in optimizer.muon.param_groups
            ]

            # For compatibility with logging/checkpoint inspection.
            self.base_lrs = self.base_muon_lrs + self.base_adamw_lrs

        else:
            self.base_lrs = [
                float(group["lr"])
                for group in optimizer.param_groups
            ]

            self.base_adamw_lrs = None
            self.base_muon_lrs = None

        # The training loop calls scheduler.step() after optimizer.step().
        # Canonical Kimi warmup therefore needs update 1 prepared up front.
        if self.prepare_first_update:
            self._set_lr(1)

    def _compute_lr(
        self,
        base_lr: float,
        min_lr: float,
        step: int,
    ) -> float:
        """
        Compute LR for a given base_lr and current step.
        """
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return base_lr * (step / max(1, self.warmup_steps))

        if self.total_steps <= self.warmup_steps:
            return min_lr

        t = min(max(step, self.warmup_steps), self.total_steps)

        denom = max(1, self.total_steps - self.warmup_steps)
        progress = (t - self.warmup_steps) / denom
        progress = min(1.0, max(0.0, progress))

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine

        return float(lr)

    def _set_lr_standard(self, step: int) -> None:
        """
        Set LR for standard torch optimizers.
        """
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            group["lr"] = self._compute_lr(
                base_lr=base_lr,
                min_lr=self.min_lr,
                step=step,
            )

    def _set_lr_hybrid(self, step: int) -> None:
        """
        Set LR for HybridMuonAdamW.

        AdamW and Muon can have different base LRs and different min LRs.
        """
        muon_min_lr = self.min_lr if self.min_muon_lr is None else self.min_muon_lr

        for i, group in enumerate(self.optimizer.adamw.param_groups):
            base_lr = self.base_adamw_lrs[i]
            group["lr"] = self._compute_lr(
                base_lr=base_lr,
                min_lr=self.min_lr,
                step=step,
            )

        for i, group in enumerate(self.optimizer.muon.param_groups):
            base_lr = self.base_muon_lrs[i]
            group["lr"] = self._compute_lr(
                base_lr=base_lr,
                min_lr=muon_min_lr,
                step=step,
            )

    def _set_lr(self, step: int) -> None:
        if self.is_hybrid:
            self._set_lr_hybrid(step)
        else:
            self._set_lr_standard(step)

    def step(self) -> None:
        """
        Advance scheduler by one step.

        Recommended order:
            optimizer.step()
            scheduler.step()
        """
        self.step_num += 1
        self._set_lr(
            self.step_num + 1 if self.prepare_first_update else self.step_num
        )

    def set_step(self, step: int) -> None:
        """
        Explicitly set scheduler step.

        Useful when resuming or manually syncing global_step.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        self.step_num = int(step)
        self._set_lr(
            self.step_num + 1 if self.prepare_first_update else self.step_num
        )

    def get_last_lr(self) -> List[float]:
        """
        Return current learning rates.
        """
        if self.is_hybrid:
            return (
                [group["lr"] for group in self.optimizer.muon.param_groups]
                + [group["lr"] for group in self.optimizer.adamw.param_groups]
            )

        return [group["lr"] for group in self.optimizer.param_groups]

    def get_lr_dict(self) -> Dict[str, Any]:
        """
        Logging-friendly LR dictionary.
        """
        if self.is_hybrid:
            muon_lrs = [group["lr"] for group in self.optimizer.muon.param_groups]
            adamw_lrs = [group["lr"] for group in self.optimizer.adamw.param_groups]

            return {
                "step": int(self.step_num),
                "muon_lr": float(muon_lrs[0]) if muon_lrs else None,
                "adamw_lr": float(adamw_lrs[0]) if adamw_lrs else None,
                "muon_lrs": [float(x) for x in muon_lrs],
                "adamw_lrs": [float(x) for x in adamw_lrs],
            }

        lrs = [group["lr"] for group in self.optimizer.param_groups]

        return {
            "step": int(self.step_num),
            "lr": float(lrs[0]) if lrs else None,
            "lrs": [float(x) for x in lrs],
        }

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "step_num": int(self.step_num),
            "total_steps": int(self.total_steps),
            "warmup_steps": int(self.warmup_steps),
            "min_lr": float(self.min_lr),
            "min_muon_lr": self.min_muon_lr,
            "is_hybrid": bool(self.is_hybrid),
            "base_lrs": list(self.base_lrs),
            "prepare_first_update": self.prepare_first_update,
        }

        if self.is_hybrid:
            state["base_adamw_lrs"] = list(self.base_adamw_lrs)
            state["base_muon_lrs"] = list(self.base_muon_lrs)

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return

        self.step_num = int(state_dict.get("step_num", 0))
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.min_lr = float(state_dict.get("min_lr", self.min_lr))
        self.prepare_first_update = bool(
            state_dict.get("prepare_first_update", self.prepare_first_update)
        )

        loaded_min_muon_lr = state_dict.get("min_muon_lr", self.min_muon_lr)
        self.min_muon_lr = (
            float(loaded_min_muon_lr)
            if loaded_min_muon_lr is not None
            else None
        )

        if self.is_hybrid:
            loaded_adamw = state_dict.get("base_adamw_lrs", None)
            loaded_muon = state_dict.get("base_muon_lrs", None)

            if (
                isinstance(loaded_adamw, (list, tuple))
                and len(loaded_adamw) == len(self.optimizer.adamw.param_groups)
            ):
                self.base_adamw_lrs = [float(x) for x in loaded_adamw]

            if (
                isinstance(loaded_muon, (list, tuple))
                and len(loaded_muon) == len(self.optimizer.muon.param_groups)
            ):
                self.base_muon_lrs = [float(x) for x in loaded_muon]

            self.base_lrs = self.base_muon_lrs + self.base_adamw_lrs

        else:
            loaded_base_lrs = state_dict.get("base_lrs", None)

            if (
                isinstance(loaded_base_lrs, (list, tuple))
                and len(loaded_base_lrs) == len(self.optimizer.param_groups)
            ):
                self.base_lrs = [float(x) for x in loaded_base_lrs]

        # Restore the LR prepared for the next optimizer update.
        self._set_lr(
            self.step_num + 1 if self.prepare_first_update else self.step_num
        )

def build_warmup_cosine_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 3e-5,
    min_muon_lr: Optional[float] = None,
    prepare_first_update: bool = False,
) -> WarmupCosineLR:
    """
    Build scheduler compatible with AdamW or HybridMuonAdamW.
    """
    return WarmupCosineLR(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        min_muon_lr=min_muon_lr,
        prepare_first_update=prepare_first_update,
    )
