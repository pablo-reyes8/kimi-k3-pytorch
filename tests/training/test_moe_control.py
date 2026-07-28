import pytest
import torch

from src.stable_latent_moe import StableLatentMoE, StableLatentMoEConfig
from training import MoEController


def tiny_moe(backend="exact"):
    return StableLatentMoE(
        StableLatentMoEConfig(
            d_model=8,
            latent_dim=4,
            num_shared_experts=1,
            num_routed_experts=4,
            routed_experts_per_token=2,
            shared_expert_hidden_dim=12,
            routed_expert_hidden_dim=8,
            quantile_backend=backend,
            histogram_num_bins=32,
            histogram_min_margin=-1,
            histogram_max_margin=1,
        )
    ).train()


@pytest.mark.parametrize("backend", ["exact", "histogram"])
def test_qb_bias_is_constant_inside_window_and_committed_afterward(backend):
    torch.manual_seed(8)
    model = tiny_moe(backend)
    controller = MoEController(model)
    initial = model.routing_bias.clone()
    controller.begin()
    model(torch.randn(2, 3, 8))
    torch.testing.assert_close(model.routing_bias, initial)
    model(torch.randn(1, 3, 8) + 0.4)
    torch.testing.assert_close(model.routing_bias, initial)
    updates = controller.commit()
    assert len(updates) == 1
    torch.testing.assert_close(model.routing_bias, updates[0].next_bias)
    assert torch.isfinite(model.routing_bias).all()
    controller.assert_clean()


def test_qb_discard_preserves_bias_and_clears_pending_state():
    model = tiny_moe("exact")
    controller = MoEController(model)
    initial = model.routing_bias.clone()
    controller.begin()
    model(torch.randn(2, 2, 8))
    controller.discard()
    torch.testing.assert_close(model.routing_bias, initial)
    assert not model._balance_accumulating
    assert model._balance_exact_scores == []


def test_qb_controller_rejects_nested_windows():
    controller = MoEController(tiny_moe())
    controller.begin()
    with pytest.raises(RuntimeError, match="already open"):
        controller.begin()
    controller.discard()
