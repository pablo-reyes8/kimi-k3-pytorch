# ============================================================
# AMP / device / precision helpers
# ============================================================

from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Optional, Union, Tuple

import torch


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "none": torch.float32,
}


def resolve_device(device: Union[str, torch.device] = "auto") -> torch.device:
    if isinstance(device, torch.device):
        requested = device
    elif device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    else:
        requested = torch.device(device)

    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but CUDA is not available.")

    if requested.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but MPS is not available.")

    return requested


def normalize_device_type(device: Union[str, torch.device] = "cuda") -> str:
    return torch.device(device).type


def resolve_amp_dtype(
    amp_dtype: str = "bf16",
    device: Union[str, torch.device] = "cuda",
) -> torch.dtype:
    amp_dtype = amp_dtype.lower()

    if amp_dtype not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported amp_dtype={amp_dtype}. "
            f"Expected one of {sorted(DTYPE_MAP.keys())}."
        )

    return DTYPE_MAP[amp_dtype]


def cuda_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False

    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            pass

    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def get_effective_amp_dtype(
    amp_dtype: str = "bf16",
    device: Union[str, torch.device] = "cuda",
    fallback_bf16_to_fp16: bool = True,
) -> Optional[torch.dtype]:
    device_type = normalize_device_type(device)
    requested_dtype = resolve_amp_dtype(amp_dtype, device=device)

    if requested_dtype == torch.float32:
        return None

    if device_type == "cuda":
        if not torch.cuda.is_available():
            return None

        if requested_dtype == torch.bfloat16:
            if cuda_supports_bf16():
                return torch.bfloat16
            return torch.float16 if fallback_bf16_to_fp16 else None

        if requested_dtype == torch.float16:
            return torch.float16

        return None

    if device_type == "cpu":
        if requested_dtype == torch.bfloat16:
            return torch.bfloat16
        return None

    return None


def should_use_grad_scaler(
    device: Union[str, torch.device] = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    fallback_bf16_to_fp16: bool = True,
) -> bool:
    if not amp_enabled:
        return False

    if normalize_device_type(device) != "cuda":
        return False

    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=device,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    return effective_dtype == torch.float16


def make_grad_scaler(
    device: Union[str, torch.device] = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    fallback_bf16_to_fp16: bool = True,
):
    enabled = should_use_grad_scaler(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    if not enabled:
        return None

    device_type = normalize_device_type(device)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            sig = inspect.signature(torch.amp.GradScaler)

            if "device" in sig.parameters:
                return torch.amp.GradScaler(device=device_type, enabled=True)

            return torch.amp.GradScaler(device_type, enabled=True)

        except Exception:
            pass

    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=True)

    return None


@contextmanager
def autocast_ctx(
    device: Union[str, torch.device] = "cuda",
    enabled: bool = True,
    amp_dtype: str = "bf16",
    cache_enabled: bool = True,
    fallback_bf16_to_fp16: bool = True,
):
    """
    Safe autocast context.

    Important:
    Do NOT wrap the yield itself in try/except.
    Otherwise, if the model forward fails, contextlib can raise:
        RuntimeError: generator didn't stop after throw()
    """
    if not enabled:
        with nullcontext():
            yield
        return

    device_type = normalize_device_type(device)

    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=device,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    if effective_dtype is None:
        with nullcontext():
            yield
        return

    if not hasattr(torch, "amp") or not hasattr(torch.amp, "autocast"):
        with nullcontext():
            yield
        return

    if device_type in {"cuda", "cpu"}:
        ctx = torch.amp.autocast(
            device_type=device_type,
            dtype=effective_dtype,
            cache_enabled=cache_enabled,
        )

        with ctx:
            yield

        return

    with nullcontext():
        yield


def setup_device_and_precision(
    device: Union[str, torch.device] = "auto",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    cache_enabled: bool = True,
    fallback_bf16_to_fp16: bool = True,
) -> Dict[str, Any]:
    resolved_device = resolve_device(device)
    device_type = normalize_device_type(resolved_device)

    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=resolved_device,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    final_amp_enabled = bool(amp_enabled and effective_dtype is not None)

    scaler = make_grad_scaler(
        device=resolved_device,
        amp_enabled=final_amp_enabled,
        amp_dtype=amp_dtype,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    return {
        "device": resolved_device,
        "device_type": device_type,
        "amp_enabled": final_amp_enabled,
        "amp_dtype_requested": amp_dtype,
        "amp_dtype_effective": effective_dtype,
        "use_grad_scaler": scaler is not None,
        "scaler": scaler,
        "cache_enabled": cache_enabled,
        "fallback_bf16_to_fp16": fallback_bf16_to_fp16,
    }


def move_batch_to_device(
    batch: Union[Dict[str, Any], Tuple[Any, ...], torch.Tensor],
    device: torch.device,
    non_blocking: bool = True,
) -> Union[Dict[str, Any], Tuple[Any, ...], torch.Tensor]:
    """
    Move a batch to device.

    Designed for batches like:

        {
            "input_ids": Tensor[B, T],
            "labels": Tensor[B, T],
            "attention_mask": optional Tensor[B, T],
            optional auxiliary tensors,
            ...
        }

    But it is recursive, so it also handles nested dicts/lists/tuples.

    Args:
        batch:
            Tensor, dict, tuple or list containing tensors.
        device:
            Destination device.
        non_blocking:
            Passed to tensor.to(...).

    Returns:
        Batch with all tensors moved to device.
    """
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=non_blocking)

    if isinstance(batch, dict):
        return {
            key: move_batch_to_device(value, device, non_blocking=non_blocking)
            for key, value in batch.items()
        }

    if isinstance(batch, tuple):
        return tuple(
            move_batch_to_device(value, device, non_blocking=non_blocking)
            for value in batch
        )

    if isinstance(batch, list):
        return [
            move_batch_to_device(value, device, non_blocking=non_blocking)
            for value in batch
        ]

    # Non-tensor metadata is left unchanged.
    return batch
