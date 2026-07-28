import torch
import torch.nn.functional as F

from src.mtp import combine_ntp_mtp_losses, masked_mtp_cross_entropy

from .conftest import tiny_mtp_head


def test_forward_shapes_loss_and_optional_outputs():
    head = tiny_mtp_head()
    ids = torch.randint(0, 23, (2, 7))
    hidden = torch.randn(2, 7, 8)
    output = head(
        hidden,
        ids,
        labels=ids,
        return_hidden_states=True,
        return_diagnostics=True,
    )
    assert output.logits.shape == (2, 5, 23)
    assert output.hidden_states.shape == (2, 5, 8)
    assert output.loss.ndim == 0 and torch.isfinite(output.loss)
    assert output.diagnostics.valid_token_count.item() == 10


def test_eval_without_labels_returns_logits_but_no_loss():
    output = tiny_mtp_head()(torch.randn(1, 5, 8), torch.arange(5).view(1, 5))
    assert output.logits.shape == (1, 3, 23)
    assert output.loss is None


def test_return_logits_false_still_computes_loss():
    head = tiny_mtp_head()
    ids = torch.randint(0, 23, (1, 5))
    output = head(torch.randn(1, 5, 8), ids, labels=ids, return_logits=False)
    assert output.logits is None
    assert torch.isfinite(output.loss)


def test_masked_cross_entropy_matches_manual_selection():
    logits = torch.randn(2, 4, 7, requires_grad=True)
    targets = torch.randint(0, 7, (2, 4))
    valid = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.bool)
    actual = masked_mtp_cross_entropy(logits, targets, valid)
    expected = F.cross_entropy(logits[valid].float(), targets[valid])
    torch.testing.assert_close(actual, expected)


def test_ignored_logits_do_not_change_loss():
    logits = torch.randn(1, 4, 6)
    targets = torch.tensor([[1, 2, 3, 4]])
    valid = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    changed = logits.clone()
    changed[:, 1::2] = 1000 * torch.randn_like(changed[:, 1::2])
    torch.testing.assert_close(
        masked_mtp_cross_entropy(logits, targets, valid),
        masked_mtp_cross_entropy(changed, targets, valid),
    )


def test_zero_valid_loss_is_finite_zero_and_graph_compatible():
    logits = torch.randn(2, 0, 5, requires_grad=True)
    loss = masked_mtp_cross_entropy(
        logits,
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(2, 0, dtype=torch.bool),
    )
    assert loss.item() == 0.0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_total_loss_formula_is_exact():
    ntp = torch.tensor(2.0)
    mtp = torch.tensor(3.0)
    torch.testing.assert_close(
        combine_ntp_mtp_losses(ntp, mtp, 0.1),
        ntp + 0.1 * mtp,
        rtol=0,
        atol=0,
    )
    assert combine_ntp_mtp_losses(ntp, None, 0.1) is ntp


def test_target_token_perturbation_changes_target_not_logits():
    head = tiny_mtp_head().eval()
    hidden = torch.randn(1, 7, 8)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    changed = ids.clone()
    changed[0, 2] = 9
    with torch.no_grad():
        left = head(hidden, ids, labels=ids)
        right = head(hidden, changed, labels=changed)
    torch.testing.assert_close(left.logits[:, 0], right.logits[:, 0])
    assert left.training_view.target_ids[0, 0] != right.training_view.target_ids[0, 0]


def test_future_tokens_cannot_change_earlier_mtp_logits():
    head = tiny_mtp_head().eval()
    hidden = torch.randn(1, 8, 8)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    changed = ids.clone()
    changed[:, 4:] = torch.tensor([[9, 10, 11, 12]])
    with torch.no_grad():
        left = head(hidden, ids).logits
        right = head(hidden, changed).logits
    torch.testing.assert_close(left[:, :3], right[:, :3], atol=1e-6, rtol=1e-5)


def test_intermediate_token_can_change_current_logit():
    head = tiny_mtp_head().eval()
    hidden = torch.randn(1, 6, 8)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = ids.clone()
    changed[0, 1] = 9
    with torch.no_grad():
        difference = (
            head(hidden, ids).logits[:, 0]
            - head(hidden, changed).logits[:, 0]
        ).abs().max()
    assert difference > 1e-7


def test_diagnostics_are_detached_and_do_not_change_logits():
    head = tiny_mtp_head().eval()
    hidden = torch.randn(2, 6, 8)
    ids = torch.randint(0, 23, (2, 6))
    with torch.no_grad():
        plain = head(hidden, ids)
        diagnosed = head(hidden, ids, return_diagnostics=True)
    torch.testing.assert_close(plain.logits, diagnosed.logits)
    assert not diagnosed.diagnostics.token_accuracy.requires_grad
    assert not diagnosed.diagnostics.mean_logit_entropy.requires_grad


def test_left_and_right_padding_compact_to_same_valid_computation():
    head = tiny_mtp_head().eval()
    core_ids = torch.tensor([3, 4, 5, 6, 7])
    core_hidden = torch.randn(5, 8)
    right_ids = torch.cat((core_ids, torch.tensor([0, 0]))).view(1, -1)
    left_ids = torch.cat((torch.tensor([0, 0]), core_ids)).view(1, -1)
    right_hidden = torch.cat((core_hidden, torch.zeros(2, 8))).unsqueeze(0)
    left_hidden = torch.cat((torch.zeros(2, 8), core_hidden)).unsqueeze(0)
    right_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    left_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    with torch.no_grad():
        right = head(right_hidden, right_ids, right_mask).logits[0, :3]
        left = head(left_hidden, left_ids, left_mask).logits[0, 2:]
    torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_packed_execution_is_rejected_instead_of_leaking():
    head = tiny_mtp_head()
    ids = torch.randint(0, 23, (1, 7))
    segments = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    try:
        head(torch.randn(1, 7, 8), ids, segment_ids=segments)
    except ValueError as error:
        assert "cross-segment" in str(error)
    else:
        raise AssertionError("packed MTP execution must be rejected")
