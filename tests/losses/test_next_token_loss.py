import pytest
import torch
import torch.nn.functional as F

from src.loss import NextTokenCrossEntropyLoss, gather_token_logprobs


def test_ntp_matches_manual_shift_and_accounting():
    torch.manual_seed(1)
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.randint(0, 7, (2, 5))
    output = NextTokenCrossEntropyLoss()(logits, labels, return_per_token=True)
    reference = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, 7), labels[:, 1:].reshape(-1)
    )
    torch.testing.assert_close(output.loss, reference)
    torch.testing.assert_close(output.loss_sum, reference * 8)
    assert output.normalizer.item() == 8
    assert output.num_valid_tokens.item() == 8


def test_ntp_already_aligned_does_not_shift():
    logits = torch.tensor([[[5.0, -1.0], [-2.0, 4.0]]])
    labels = torch.tensor([[0, 1]])
    output = NextTokenCrossEntropyLoss(alignment="already_aligned")(logits, labels)
    torch.testing.assert_close(
        output.loss, F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
    )


def test_ntp_masks_padding_loss_and_boundaries():
    logits = torch.randn(1, 6, 5)
    labels = torch.tensor([[0, 1, 2, 3, 4, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    loss_mask = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.bool)
    boundary = torch.tensor([[1, 1, 1, 0, 1, 1]], dtype=torch.bool)
    output = NextTokenCrossEntropyLoss()(
        logits,
        labels,
        attention_mask=attention,
        loss_mask=loss_mask,
        boundary_mask=boundary,
    )
    expected = F.cross_entropy(logits[:, 0].float(), labels[:, 1])
    torch.testing.assert_close(output.loss, expected)
    assert output.num_valid_tokens.item() == 1


def test_ntp_visual_placeholder_target_can_be_ignored():
    logits = torch.randn(1, 5, 8)
    labels = torch.tensor([[1, -100, -100, 4, 5]])
    changed = logits.clone()
    changed[:, :2] = 1000 * torch.randn_like(changed[:, :2])
    criterion = NextTokenCrossEntropyLoss()
    # Logit position 0 predicts ignored label 1; position 1 predicts ignored label 2.
    torch.testing.assert_close(criterion(logits, labels).loss, criterion(changed, labels).loss)


def test_ntp_token_weights_define_the_normalizer():
    logits = torch.randn(1, 4, 6)
    labels = torch.randint(0, 6, (1, 4))
    weights = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    output = NextTokenCrossEntropyLoss()(logits, labels, token_weights=weights)
    per = F.cross_entropy(
        logits[:, :-1].reshape(-1, 6), labels[:, 1:].reshape(-1), reduction="none"
    )
    torch.testing.assert_close(output.loss, (per * weights[:, 1:].flatten()).sum() / 6)
    assert output.normalizer.item() == 6


def test_ntp_zero_valid_policy_and_connected_gradient():
    logits = torch.randn(1, 3, 4, requires_grad=True)
    labels = torch.full((1, 3), -100)
    with pytest.raises(ValueError, match="no valid"):
        NextTokenCrossEntropyLoss()(logits, labels)
    output = NextTokenCrossEntropyLoss(zero_valid_policy="connected_zero")(logits, labels)
    output.loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_ntp_bf16_computes_fp32_and_has_finite_gradient():
    logits = torch.randn(2, 4, 9, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 9, (2, 4))
    output = NextTokenCrossEntropyLoss()(logits, labels)
    assert output.loss.dtype == torch.float32
    output.loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_ntp_rejects_invalid_targets_and_negative_weights():
    logits = torch.randn(1, 3, 4)
    with pytest.raises(ValueError, match="outside"):
        NextTokenCrossEntropyLoss()(logits, torch.tensor([[0, 8, 1]]))
    with pytest.raises(ValueError, match="non-negative"):
        NextTokenCrossEntropyLoss()(
            logits, torch.tensor([[0, 1, 2]]), token_weights=torch.tensor([[1.0, -1.0, 1.0]])
        )


def test_ntp_nonfinite_logits_at_ignored_positions_do_not_poison_reduction():
    logits = torch.randn(1, 4, 5)
    logits[:, 0] = float("nan")
    labels = torch.tensor([[0, -100, 2, 3]])
    output = NextTokenCrossEntropyLoss()(logits, labels)
    assert torch.isfinite(output.loss)


def test_ntp_gradient_matches_double_precision_reference():
    torch.manual_seed(2)
    logits = torch.randn(1, 4, 5, dtype=torch.double, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 3]])
    output = NextTokenCrossEntropyLoss()(logits, labels)
    reference = F.cross_entropy(logits[:, :-1].float().reshape(-1, 5), labels[:, 1:].reshape(-1))
    grad, = torch.autograd.grad(output.loss, logits, retain_graph=True)
    reference_grad, = torch.autograd.grad(reference, logits)
    torch.testing.assert_close(grad, reference_grad)


def test_gather_token_logprobs_matches_reference_and_validates():
    logits = torch.randn(2, 3, 5, dtype=torch.bfloat16)
    ids = torch.randint(0, 5, (2, 3))
    actual = gather_token_logprobs(logits, ids, temperature=0.7)
    expected = F.log_softmax(logits.float() / 0.7, -1).gather(-1, ids[..., None]).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
    with pytest.raises(ValueError, match="vocabulary"):
        gather_token_logprobs(logits, torch.full_like(ids, 9))
