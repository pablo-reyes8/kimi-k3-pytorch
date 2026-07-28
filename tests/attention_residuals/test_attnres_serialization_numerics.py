import copy
import io

import pytest
import torch

from src.hybrid_backbone import HybridAttentionBackbone
from tests.attention_residuals.conftest import (
    activate_depth_queries,
    attnres_backbone,
    backbone_config,
)


@pytest.mark.parametrize(
    "depth_mode,backend",
    [("full", "eager"), ("block", "eager"), ("block", "two_phase")],
)
def test_state_dict_roundtrip_exact_eval_and_query_preservation(
    depth_mode, backend
):
    source = attnres_backbone(
        depth_mode=depth_mode, backend=backend
    ).double().eval()
    activate_depth_queries(source)
    target = HybridAttentionBackbone(
        backbone_config(depth_mode, backend)
    ).double().eval()
    target.load_state_dict(source.state_dict())
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    torch.testing.assert_close(
        source(x).last_hidden_state,
        target(x).last_hidden_state,
        rtol=0,
        atol=0,
    )
    for left, right in zip(
        source.depth_site_metadata, target.depth_site_metadata
    ):
        assert left == right
    torch.testing.assert_close(
        source.final_output_attnres.pseudo_query,
        target.final_output_attnres.pseudo_query,
    )


def test_zero_queries_survive_serialization():
    model = attnres_backbone(depth_mode="block")
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state_dict = torch.load(buffer, weights_only=True)
    query_tensors = [
        value for name, value in state_dict.items()
        if name.endswith("pseudo_query")
    ]
    assert len(query_tensors) == 2 * len(model.layers) + 1
    assert all(torch.count_nonzero(value) == 0 for value in query_tensors)


def test_deterministic_eval_and_dropout_disabled():
    model = attnres_backbone(
        depth_mode="block",
        residual_dropout=0.7,
        ffn_dropout=0.7,
    ).eval()
    x = torch.randn(2, 6, 8)
    torch.testing.assert_close(
        model(x).last_hidden_state,
        model(x).last_hidden_state,
        rtol=0,
        atol=0,
    )


def test_to_float64_moves_all_attnres_parameters():
    model = attnres_backbone(depth_mode="block").double()
    for name, parameter in model.named_parameters():
        if "attnres" in name:
            assert parameter.dtype == torch.float64


@pytest.mark.parametrize(
    "depth_mode,backend",
    [("full", "eager"), ("block", "eager"), ("block", "two_phase")],
)
def test_fp32_large_and_small_sources_outputs_gradients_are_finite(
    depth_mode, backend
):
    model = attnres_backbone(
        depth_mode=depth_mode, backend=backend
    )
    activate_depth_queries(model)
    x = torch.randn(2, 9, 8)
    x[0] *= 1e4
    x[1] *= 1e-4
    x.requires_grad_()
    output = model(x, output_depth_weights=True)
    output.last_hidden_state.square().mean().backward()
    assert torch.isfinite(output.last_hidden_state).all()
    assert torch.isfinite(output.depth_outputs.averaged_weight_matrix).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "depth_mode,backend",
    [("full", "eager"), ("block", "eager"), ("block", "two_phase")],
)
def test_cpu_bfloat16_full_prefill_decode(depth_mode, backend):
    model = attnres_backbone(
        depth_mode=depth_mode, backend=backend
    ).bfloat16().eval()
    x = torch.randn(2, 5, 8).bfloat16()
    prefill = model(x[:, :4], mode="prefill", use_cache=True)
    decode = model(
        x[:, 4:],
        cache=prefill.cache,
        mode="decode",
        use_cache=True,
    )
    assert prefill.last_hidden_state.dtype == torch.bfloat16
    assert decode.last_hidden_state.dtype == torch.bfloat16
    assert torch.isfinite(prefill.last_hidden_state.float()).all()
    assert torch.isfinite(decode.last_hidden_state.float()).all()


def test_ephemeral_sources_are_not_in_state_dict_or_module_attributes():
    model = attnres_backbone(depth_mode="block").eval()
    initial_keys = tuple(model.state_dict())
    for _ in range(5):
        model(torch.randn(2, 1, 8))
    assert tuple(model.state_dict()) == initial_keys
    forbidden = {"sources", "completed_blocks", "partial_block", "phase_stats"}
    for module in model.modules():
        assert forbidden.isdisjoint(vars(module))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "depth_mode,backend",
    [("full", "eager"), ("block", "eager"), ("block", "two_phase")],
)
def test_cuda_bfloat16_against_fp32(depth_mode, backend):
    fp32 = attnres_backbone(
        depth_mode=depth_mode, backend=backend
    ).cuda().eval()
    activate_depth_queries(fp32)
    bf16 = copy.deepcopy(fp32).bfloat16()
    x = torch.randn(2, 6, 8, device="cuda")
    reference = fp32(x).last_hidden_state
    actual = bf16(x.bfloat16()).last_hidden_state.float()
    torch.testing.assert_close(actual, reference, rtol=0.08, atol=0.03)
