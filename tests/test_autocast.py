import pytest
import torch

from training.autocast import (
    autocast_ctx,
    get_effective_amp_dtype,
    make_grad_scaler,
    move_batch_to_device,
    normalize_device_type,
    resolve_amp_dtype,
    resolve_device,
    setup_device_and_precision,
    should_use_grad_scaler,
)


def test_resolve_cpu_auto_and_device_object():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
    assert normalize_device_type(torch.device("cpu")) == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires CUDA to be absent")
def test_unavailable_cuda_request_rejected():
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("fp16", torch.float16),
        ("float16", torch.float16),
        ("fp32", torch.float32),
        ("none", torch.float32),
    ],
)
def test_amp_dtype_aliases(name, expected):
    assert resolve_amp_dtype(name, "cpu") == expected


def test_unknown_amp_dtype_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_amp_dtype("tf32")


@pytest.mark.parametrize(
    "name,expected",
    [("bf16", torch.bfloat16), ("fp16", None), ("fp32", None)],
)
def test_effective_cpu_amp_policy(name, expected):
    assert get_effective_amp_dtype(name, "cpu") == expected


def test_cpu_never_uses_grad_scaler():
    assert not should_use_grad_scaler("cpu", amp_enabled=True, amp_dtype="bf16")
    assert make_grad_scaler("cpu", amp_enabled=True, amp_dtype="bf16") is None


def test_disabled_autocast_is_exact_fp32_and_propagates_exceptions():
    x = torch.randn(4, 4)
    with autocast_ctx("cpu", enabled=False):
        y = x @ x
    assert y.dtype == torch.float32
    with pytest.raises(RuntimeError, match="sentinel"):
        with autocast_ctx("cpu", enabled=False):
            raise RuntimeError("sentinel")


def test_cpu_bfloat16_autocast_changes_eligible_operation_dtype():
    x = torch.randn(16, 16)
    with autocast_ctx("cpu", enabled=True, amp_dtype="bf16"):
        y = x @ x
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()


def test_precision_setup_reports_effective_cpu_contract():
    disabled = setup_device_and_precision("cpu", amp_enabled=False)
    assert disabled["device"] == torch.device("cpu")
    assert disabled["amp_enabled"] is False
    assert disabled["scaler"] is None
    enabled = setup_device_and_precision("cpu", amp_enabled=True, amp_dtype="bf16")
    assert enabled["amp_enabled"] is True
    assert enabled["amp_dtype_effective"] == torch.bfloat16
    assert enabled["use_grad_scaler"] is False


def test_move_batch_to_device_recurses_and_preserves_container_types_metadata():
    batch = {
        "tensor": torch.ones(2),
        "tuple": (torch.zeros(1), "kept"),
        "list": [torch.arange(2), 7],
        "dict": {"nested": torch.tensor(3)},
    }
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert isinstance(moved, dict)
    assert isinstance(moved["tuple"], tuple)
    assert isinstance(moved["list"], list)
    assert moved["tuple"][1] == "kept" and moved["list"][1] == 7
    assert all(
        tensor.device.type == "cpu"
        for tensor in (
            moved["tensor"],
            moved["tuple"][0],
            moved["list"][0],
            moved["dict"]["nested"],
        )
    )
