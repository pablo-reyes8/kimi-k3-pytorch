import pytest
import torch
import torch.nn.functional as F

from src.loss import MultiTokenPredictionLoss


def test_mtp_single_depth_matches_manual_ce():
    logits = torch.randn(2, 4, 7, requires_grad=True)
    labels = torch.randint(0, 7, (2, 4))
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.bool)
    output = MultiTokenPredictionLoss()(logits, labels, mtp_loss_mask=mask)
    reference = F.cross_entropy(logits[mask].float(), labels[mask])
    torch.testing.assert_close(output.loss, reference)
    assert output.normalizer.item() == 6
    assert output.future_offsets == (2,)


def test_mtp_multi_depth_weights_apply_to_sums_and_counts():
    logits = torch.randn(1, 2, 3, 5)
    labels = torch.randint(0, 5, (1, 2, 3))
    mask = torch.tensor([[[1, 1, 1], [1, 0, 0]]], dtype=torch.bool)
    output = MultiTokenPredictionLoss(depth_weights=(1.0, 3.0))(
        logits, labels, mtp_loss_mask=mask, future_offsets=(2, 3), return_per_depth=True
    )
    per = F.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1), reduction="none").reshape(1, 2, 3)
    expected_sum = per[0, 0].sum() + 3 * per[0, 1, 0]
    torch.testing.assert_close(output.loss_sum, expected_sum)
    assert output.normalizer.item() == 6
    assert output.future_offsets == (2, 3)


def test_mtp_short_or_empty_targets_return_connected_zero():
    logits = torch.randn(1, 0, 5, requires_grad=True)
    labels = torch.empty(1, 0, dtype=torch.long)
    output = MultiTokenPredictionLoss()(logits, labels)
    output.loss.backward()
    assert logits.grad is not None
    assert output.loss.item() == 0


def test_mtp_consumes_mask_without_rebuilding_alignment():
    logits = torch.randn(1, 3, 4)
    labels = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[False, True, False]])
    output = MultiTokenPredictionLoss()(logits, labels, mtp_loss_mask=mask)
    torch.testing.assert_close(output.loss, F.cross_entropy(logits[:, 1], labels[:, 1]))


def test_mtp_ignores_phase9_ignore_targets():
    logits = torch.randn(1, 3, 4)
    labels = torch.tensor([[-100, 2, -100]])
    output = MultiTokenPredictionLoss()(logits, labels)
    torch.testing.assert_close(output.loss, F.cross_entropy(logits[:, 1], labels[:, 1]))
    assert output.num_valid_tokens.item() == 1


def test_mtp_bf16_is_fp32_and_backpropagates():
    logits = torch.randn(1, 3, 6, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 6, (1, 3))
    output = MultiTokenPredictionLoss()(logits, labels)
    assert output.loss.dtype == torch.float32
    output.loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_mtp_validates_depth_metadata_and_weights():
    logits = torch.randn(1, 2, 3, 5)
    labels = torch.randint(0, 5, (1, 2, 3))
    with pytest.raises(ValueError, match="future_offsets"):
        MultiTokenPredictionLoss()(logits, labels, future_offsets=(2,))
    with pytest.raises(ValueError, match="depth_weights"):
        MultiTokenPredictionLoss(depth_weights=(1.0,))(logits, labels)

