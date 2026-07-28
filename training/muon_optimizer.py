# ============================================================
# MUON + HYBRID MUON-ADAMW OPTIMIZER
# Generic Muon + AdamW training utilities
# ============================================================

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# ============================================================
# Newton-Schulz orthogonalization
# ============================================================

@torch.no_grad()
def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Approximate zeroth power / orthogonalized update using Newton-Schulz iterations.

    Args:
        G:
            2D gradient/update matrix.
        steps:
            Number of Newton-Schulz iterations.
        eps:
            Numerical stability epsilon.

    Returns:
        Tensor with same shape/device/dtype as G.
    """
    if G.dim() != 2:
        raise ValueError(f"zeropower_via_newtonschulz5 expects a 2D tensor, got shape={tuple(G.shape)}.")

    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}.")

    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}.")

    original_dtype = G.dtype
    original_device = G.device

    X = G.detach().float()

    if not torch.isfinite(X).all():
        raise FloatingPointError("Input to zeropower_via_newtonschulz5 contains NaN or Inf.")

    norm = X.norm()

    if norm <= eps:
        return torch.zeros_like(G)

    # For rectangular matrices, operate on the cheaper orientation.
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    X = X / (X.norm() + eps)

    # Quintic coefficients commonly used in modern Muon-style implementations.
    a = 3.4445
    b = -4.7750
    c = 2.0315

    for _ in range(steps):
        A = X @ X.T
        B = A @ A
        X = a * X + b * (A @ X) + c * (B @ X)

    if transposed:
        X = X.T

    return X.to(device=original_device, dtype=original_dtype)


# ============================================================
# Muon optimizer
# ============================================================

class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for 2D matrix parameters.

    Intended use:
        - hidden Linear weights
        - attention projections
        - eligible dense projection matrices
        - selected internal 2D transformations

    Not intended for:
        - embeddings
        - LM heads
        - norms
        - biases
        - scalar/vector parameters
        - static/gating parameters
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}.")

        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must satisfy 0 <= momentum < 1, got {momentum}.")

        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}.")

        if ns_steps <= 0:
            raise ValueError(f"ns_steps must be > 0, got {ns_steps}.")

        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
        )

        super().__init__(params, defaults)

        # Validate all parameters upfront.
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "Muon only supports 2D parameters. "
                        f"Got parameter with shape={tuple(p.shape)}."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform one Muon optimization step.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.ndim != 2:
                    raise ValueError(
                        "Muon only supports 2D parameters during step. "
                        f"Got shape={tuple(p.shape)}."
                    )

                grad = p.grad.detach()

                if not torch.isfinite(grad).all():
                    raise FloatingPointError(
                        f"Non-finite gradient found in Muon parameter with shape={tuple(p.shape)}."
                    )

                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]

                # Momentum buffer tracks raw gradient/update direction.
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf

                update = zeropower_via_newtonschulz5(
                    update,
                    steps=ns_steps,
                    eps=eps,
                )

                if weight_decay != 0.0:
                    # Decoupled weight decay.
                    p.data.mul_(1.0 - lr * weight_decay)

                p.data.add_(update, alpha=-lr)

        return loss


# ============================================================
# Hybrid optimizer wrapper
# ============================================================

class HybridMuonAdamW:
    """
    Thin wrapper combining:
        - Muon for selected 2D hidden matrices
        - AdamW for all remaining parameters

    It exposes:
        - zero_grad
        - step
        - state_dict/load_state_dict
        - set_lr
        - param_groups

    Compatible with our checkpointing code.
    """

    def __init__(
        self,
        muon_optimizer: Muon,
        adamw_optimizer: torch.optim.AdamW,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.muon = muon_optimizer
        self.adamw = adamw_optimizer
        self.metadata = metadata or {}

    @property
    def param_groups(self):
        # Returned list is for inspection/logging.
        # For changing lr, use set_lr().
        return self.muon.param_groups + self.adamw.param_groups

    @property
    def defaults(self):
        return {
            "muon": getattr(self.muon, "defaults", {}),
            "adamw": getattr(self.adamw, "defaults", {}),
        }

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        self.muon.step()
        self.adamw.step()

        return loss

    def set_lr(self, lr: float, muon_lr: Optional[float] = None):
        """
        Scheduler-compatible LR setter.

        Args:
            lr:
                AdamW LR.
            muon_lr:
                Optional Muon LR. If None, uses lr.
        """
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}.")

        if muon_lr is not None and muon_lr <= 0:
            raise ValueError(f"muon_lr must be > 0, got {muon_lr}.")

        for group in self.adamw.param_groups:
            group["lr"] = lr

        effective_muon_lr = lr if muon_lr is None else muon_lr

        for group in self.muon.param_groups:
            group["lr"] = effective_muon_lr

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
            "metadata": self.metadata,
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.metadata = state_dict.get("metadata", self.metadata)


# ============================================================
# Parameter grouping helpers
# ============================================================

def _get_module_by_parameter_name(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Build mapping from full parameter name to the owning module.
    """
    param_to_module: Dict[str, nn.Module] = {}

    for module_name, module in model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            param_to_module[full_name] = module

    return param_to_module


def _is_embedding_parameter(name: str, module: Optional[nn.Module] = None) -> bool:
    lower = name.lower()

    if "embedding" in lower or "embed" in lower or "wte" in lower:
        return True

    if module is not None and isinstance(module, nn.Embedding):
        return True

    return False


def _is_norm_parameter(name: str, module: Optional[nn.Module] = None) -> bool:
    lower = name.lower()

    if "norm" in lower:
        return True

    if module is not None:
        norm_classes = (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LocalResponseNorm,
        )

        if isinstance(module, norm_classes):
            return True

    return False


def should_use_muon(
    name: str,
    param: nn.Parameter,
    module: Optional[nn.Module] = None,
) -> bool:
    """
    Conservative rule for assigning a parameter to Muon.

    Muon gets only:
        - trainable
        - 2D
        - .weight
        - not excluded by known fragile/special modules
    """
    if not param.requires_grad:
        return False

    if param.ndim != 2:
        return False

    if not name.endswith(".weight"):
        return False

    lower = name.lower()

    exclude_keywords = [
        "embedding",
        "embed",
        "token_embedding",
        "lm_head",
        "prediction_head",

        # Norms and small controls.
        "norm",
        "bias",
        "static_",
        "alpha_",
        "bias_a",
        "bias_b",
        "temperature",
        "scale",

        "readout",

        # Explicit loss/gating weights.
        "depth_loss_weights",
    ]

    if any(k in lower for k in exclude_keywords):
        return False

    if _is_embedding_parameter(name, module):
        return False

    if _is_norm_parameter(name, module):
        return False

    return True


def should_use_adamw_no_decay(
    name: str,
    param: nn.Parameter,
    module: Optional[nn.Module] = None,
) -> bool:
    """
    Conservative AdamW no-decay rule.
    """
    lower = name.lower()

    if not param.requires_grad:
        return False

    if param.ndim < 2:
        return True

    if name.endswith(".bias"):
        return True

    if _is_norm_parameter(name, module):
        return True

    if _is_embedding_parameter(name, module):
        return True

    no_decay_keywords = [
        "lm_head",
        "prediction_head",
        "bias_a",
        "bias_b",
        "alpha_",
        "static_",
        "temperature",
        "scale",
        "depth_loss_weights",
        "readout",
    ]

    if any(k in lower for k in no_decay_keywords):
        return True

    return False


def build_muon_adamw_parameter_groups(
    model: nn.Module,
    adamw_weight_decay: float = 0.1,
    muon_weight_decay: float = 0.0,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build parameter groups for HybridMuonAdamW.

    Returns:
        muon_groups:
            list of groups for Muon.
        adamw_groups:
            list of groups for AdamW.
        metadata:
            diagnostics and parameter names.
    """
    if adamw_weight_decay < 0:
        raise ValueError(f"adamw_weight_decay must be >= 0, got {adamw_weight_decay}.")

    if muon_weight_decay < 0:
        raise ValueError(f"muon_weight_decay must be >= 0, got {muon_weight_decay}.")

    param_to_module = _get_module_by_parameter_name(model)

    muon_params = []
    adamw_decay_params = []
    adamw_no_decay_params = []

    muon_names = []
    adamw_decay_names = []
    adamw_no_decay_names = []

    seen_param_ids = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = param_to_module.get(name, None)

        if should_use_muon(name, param, module):
            target_list = muon_params
            target_names = muon_names
        else:
            if should_use_adamw_no_decay(name, param, module):
                target_list = adamw_no_decay_params
                target_names = adamw_no_decay_names
            else:
                target_list = adamw_decay_params
                target_names = adamw_decay_names

        pid = id(param)

        if pid in seen_param_ids:
            raise RuntimeError(f"Duplicate parameter assignment detected for parameter: {name}")

        seen_param_ids.add(pid)

        target_list.append(param)
        target_names.append(name)

    # Validate no trainable params missing.
    trainable_params = [
        (name, p)
        for name, p in model.named_parameters()
        if p.requires_grad
    ]

    trainable_ids = {id(p) for _, p in trainable_params}
    assigned_ids = {id(p) for p in muon_params + adamw_decay_params + adamw_no_decay_params}

    missing = [
        name
        for name, p in trainable_params
        if id(p) not in assigned_ids
    ]

    extra = assigned_ids - trainable_ids

    if missing:
        raise RuntimeError(f"Some trainable parameters were not assigned to any optimizer group: {missing[:20]}")

    if extra:
        raise RuntimeError("Some assigned parameters are not trainable model parameters.")

    # Muon must only receive 2D tensors.
    bad_muon = [
        name
        for name, p in zip(muon_names, muon_params)
        if p.ndim != 2
    ]

    if bad_muon:
        raise RuntimeError(f"Non-2D parameters assigned to Muon: {bad_muon[:20]}")

    muon_groups = [
        {
            "params": muon_params,
            "weight_decay": muon_weight_decay,
            "group_name": "muon",
        }
    ]

    adamw_groups = [
        {
            "params": adamw_decay_params,
            "weight_decay": adamw_weight_decay,
            "group_name": "adamw_decay",
        },
        {
            "params": adamw_no_decay_params,
            "weight_decay": 0.0,
            "group_name": "adamw_no_decay",
        },
    ]

    metadata = {
        "num_muon_tensors": len(muon_params),
        "num_adamw_decay_tensors": len(adamw_decay_params),
        "num_adamw_no_decay_tensors": len(adamw_no_decay_params),
        "num_adamw_tensors": len(adamw_decay_params) + len(adamw_no_decay_params),

        "num_muon_params": int(sum(p.numel() for p in muon_params)),
        "num_adamw_decay_params": int(sum(p.numel() for p in adamw_decay_params)),
        "num_adamw_no_decay_params": int(sum(p.numel() for p in adamw_no_decay_params)),
        "num_adamw_params": int(
            sum(p.numel() for p in adamw_decay_params)
            + sum(p.numel() for p in adamw_no_decay_params)
        ),

        "muon_names": muon_names,
        "adamw_decay_names": adamw_decay_names,
        "adamw_no_decay_names": adamw_no_decay_names,
    }

    metadata["total_trainable_params"] = (
        metadata["num_muon_params"]
        + metadata["num_adamw_params"]
    )

    if verbose:
        print("=" * 80)
        print("Hybrid Muon + AdamW parameter groups")
        print("=" * 80)
        print(f"Muon tensors          : {metadata['num_muon_tensors']}")
        print(f"AdamW decay tensors   : {metadata['num_adamw_decay_tensors']}")
        print(f"AdamW no-decay tensors: {metadata['num_adamw_no_decay_tensors']}")
        print("-" * 80)
        print(f"Muon params           : {metadata['num_muon_params']:,}")
        print(f"AdamW params          : {metadata['num_adamw_params']:,}")
        print(f"Total trainable params: {metadata['total_trainable_params']:,}")
        print("=" * 80)

    return muon_groups, adamw_groups, metadata


# ============================================================
# Optimizer builder
# ============================================================

def build_muon_adamw_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    muon_lr: Optional[float] = None,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    muon_eps: float = 1e-7,
    muon_weight_decay: float = 0.0,
    verbose: bool = False,
) -> Tuple[HybridMuonAdamW, Dict[str, Any]]:
    """
    Build hybrid Muon + AdamW optimizer.

    Usage:
        optimizer, opt_info = build_muon_adamw_optimizer(
            model=model,
            learning_rate=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            eps=1e-8,
            muon_lr=None,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_ns_steps=5,
            muon_weight_decay=0.0,
            verbose=True,
        )
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")

    if muon_lr is not None and muon_lr <= 0:
        raise ValueError(f"muon_lr must be > 0 if provided, got {muon_lr}.")

    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}.")

    if len(betas) != 2:
        raise ValueError(f"betas must have length 2, got {betas}.")

    effective_muon_lr = learning_rate if muon_lr is None else muon_lr

    muon_groups, adamw_groups, metadata = build_muon_adamw_parameter_groups(
        model=model,
        adamw_weight_decay=weight_decay,
        muon_weight_decay=muon_weight_decay,
        verbose=verbose,
    )

    # If no Muon params exist, this is likely a config/detection issue.
    if metadata["num_muon_tensors"] == 0:
        raise RuntimeError(
            "No parameters were assigned to Muon. "
            "Check should_use_muon exclusions or model architecture."
        )

    muon_optimizer = Muon(
        muon_groups,
        lr=effective_muon_lr,
        momentum=muon_momentum,
        weight_decay=muon_weight_decay,
        nesterov=muon_nesterov,
        ns_steps=muon_ns_steps,
        eps=muon_eps,
    )

    adamw_optimizer = torch.optim.AdamW(
        adamw_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    metadata["optimizer_type"] = "muon_adamw"
    metadata["learning_rate"] = learning_rate
    metadata["muon_lr"] = effective_muon_lr
    metadata["adamw_lr"] = learning_rate
    metadata["weight_decay"] = weight_decay
    metadata["muon_weight_decay"] = muon_weight_decay
    metadata["muon_momentum"] = muon_momentum
    metadata["muon_nesterov"] = muon_nesterov
    metadata["muon_ns_steps"] = muon_ns_steps
    metadata["muon_eps"] = muon_eps

    optimizer = HybridMuonAdamW(
        muon_optimizer=muon_optimizer,
        adamw_optimizer=adamw_optimizer,
        metadata=metadata,
    )

    return optimizer, metadata
