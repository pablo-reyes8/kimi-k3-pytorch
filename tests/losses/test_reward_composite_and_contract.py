import pytest
import torch

from src.loss import (
    KimiLossConfig,
    KimiPretrainingLoss,
    KimiRewardComposer,
    KimiTrainingObjective,
    KimiTrainingPhase,
    RewardWeights,
    apply_reasoning_budget_override,
    apply_verbosity_binary_rule,
)


def test_reward_composer_requires_explicit_weights_and_detaches():
    reward = torch.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError, match="explicit"):
        KimiRewardComposer()(
            verifier_reward=reward, generative_reward=None,
            task_specific_reward=None, weights=RewardWeights()
        )
    result = KimiRewardComposer()(
        verifier_reward=reward,
        generative_reward=torch.tensor([2.0, 4.0]),
        task_specific_reward=None,
        weights=RewardWeights(verifier=2.0, generative=-0.5),
    )
    torch.testing.assert_close(result, torch.tensor([1.0, 2.0]))
    assert not result.requires_grad


def test_budget_override_and_verbosity_rules_are_exact():
    result = apply_reasoning_budget_override(
        torch.tensor([0.5, 0.8]),
        used_tokens=torch.tensor([10, 21]),
        base_budget=torch.tensor([10, 10]),
        budget_multiplier=2.0,
    )
    torch.testing.assert_close(result, torch.tensor([0.5, -1.0]))
    wins = apply_verbosity_binary_rule(
        torch.tensor([True, True]),
        output_length=torch.tensor([10, 11]),
        initial_verbosity=torch.tensor([5, 5]),
        sigma=2.0,
    )
    assert wins.tolist() == [True, False]


def test_pretraining_composite_is_ntp_plus_lambda_mtp_only():
    logits = torch.randn(1, 4, 6)
    labels = torch.randint(0, 6, (1, 4))
    mtp_logits = torch.randn(1, 2, 6)
    mtp_labels = torch.randint(0, 6, (1, 2))
    output = KimiPretrainingLoss(lambda_mtp=0.4)(
        logits=logits,
        labels=labels,
        mtp_logits=mtp_logits,
        mtp_labels=mtp_labels,
        moe_diagnostics={"loads": torch.tensor([1, 2])},
    )
    torch.testing.assert_close(output.loss, output.ntp.loss + 0.4 * output.mtp.loss)
    assert output.moe_aux_loss is None
    assert not hasattr(output, "router_z_loss")


def test_quantile_balancing_is_not_a_loss_public_api():
    import src.loss as losses

    assert not hasattr(losses, "MoEBalancingLoss")
    assert not hasattr(losses, "RouterZLoss")


def test_global_token_weighted_reduction_matches_concatenation():
    from src.loss import NextTokenCrossEntropyLoss

    logits = torch.randn(2, 6, 7)
    labels = torch.randint(0, 7, (2, 6))
    masks = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    criterion = NextTokenCrossEntropyLoss()
    rank0 = criterion(logits[:1], labels[:1], attention_mask=masks[:1])
    rank1 = criterion(logits[1:], labels[1:], attention_mask=masks[1:])
    global_value = (rank0.loss_sum + rank1.loss_sum) / (rank0.normalizer + rank1.normalizer)
    concatenated = criterion(logits, labels, attention_mask=masks)
    torch.testing.assert_close(global_value, concatenated.loss)
    assert not torch.allclose((rank0.loss + rank1.loss) / 2, concatenated.loss)


def _config(mode="sampled_token_pg"):
    return KimiLossConfig(
        lambda_mtp=0.0,
        rl_ratio_clip_min=0.8,
        rl_ratio_clip_max=1.2,
        rl_log_ratio_l2_coef=0.1,
        mopd_reward_clip_max=2.0,
        mopd_mode=mode,
    )


def test_loss_config_roundtrip_and_unpublished_values_are_explicit():
    config = _config()
    assert KimiLossConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="unpublished"):
        KimiTrainingObjective(KimiLossConfig())


def test_training_objective_dispatches_strictly():
    objective = KimiTrainingObjective(_config())
    logits = torch.randn(1, 3, 5)
    labels = torch.randint(0, 5, (1, 3))
    result = objective(phase=KimiTrainingPhase.PRETRAIN, logits=logits, labels=labels)
    assert result.phase is KimiTrainingPhase.PRETRAIN
    torch.testing.assert_close(result.loss, result.output.loss)
    with pytest.raises(ValueError, match="another phase"):
        objective(phase="pretrain", logits=logits, labels=labels, rewards=torch.ones(1))


def test_loss_modules_have_no_parameters_or_batch_state():
    objective = KimiTrainingObjective(_config())
    assert list(objective.parameters()) == []
    assert objective.state_dict() == {}

