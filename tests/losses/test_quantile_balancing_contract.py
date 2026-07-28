import torch

from src import StableLatentMoE, StableLatentMoEConfig
from src.loss import KimiPretrainingLoss


def _moe():
    return StableLatentMoE(
        StableLatentMoEConfig(
            d_model=8,
            latent_dim=4,
            num_shared_experts=1,
            num_routed_experts=4,
            routed_experts_per_token=2,
            shared_expert_hidden_dim=12,
            routed_expert_hidden_dim=8,
        )
    )


def test_quantile_routing_bias_is_buffer_not_parameter():
    moe = _moe()
    assert "routing_bias" in dict(moe.router.named_buffers())
    assert "routing_bias" not in dict(moe.router.named_parameters())
    assert not moe.routing_bias.requires_grad


def test_loss_does_not_mutate_quantile_balancing_state():
    moe = _moe()
    before = moe.routing_bias.clone()
    objective = KimiPretrainingLoss()
    objective(logits=torch.randn(1, 4, 9), labels=torch.randint(0, 9, (1, 4)))
    torch.testing.assert_close(moe.routing_bias, before)


def test_quantile_balancing_diagnostics_are_not_loss_terms():
    diagnostics = {"expert_loads": torch.tensor([2, 1, 0, 3])}
    output = KimiPretrainingLoss()(
        logits=torch.randn(1, 4, 9),
        labels=torch.randint(0, 9, (1, 4)),
        moe_diagnostics=diagnostics,
    )
    assert output.moe_diagnostics is diagnostics
    assert output.moe_aux_loss is None
    assert all("balance" not in name for name in output.__dataclass_fields__ if name.endswith("loss"))


def test_quantile_balancing_does_not_update_during_eval():
    moe = _moe().eval()
    before = moe.routing_bias.clone()
    _ = moe(torch.randn(2, 8))
    torch.testing.assert_close(moe.routing_bias, before)

