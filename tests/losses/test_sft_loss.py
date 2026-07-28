import pytest
import torch
import torch.nn.functional as F

from src.loss import SFTComponent, SFTTrajectoryCrossEntropyLoss


def _trajectory():
    logits = torch.randn(1, 8, 11, requires_grad=True)
    labels = torch.arange(8).unsqueeze(0) % 11
    # system, user, reasoning, tool call, tool arg, observation, final, EOT
    assistant = torch.tensor([[0, 0, 1, 1, 1, 0, 1, 1]], dtype=torch.bool)
    components = torch.tensor([[
        0, 0, SFTComponent.REASONING, SFTComponent.TOOL_CALL,
        SFTComponent.TOOL_ARGUMENT, 0, SFTComponent.FINAL_ANSWER,
        SFTComponent.END_OF_TURN,
    ]])
    return logits, labels, assistant, components


def test_sft_matches_manual_assistant_only_cross_entropy():
    logits, labels, assistant, components = _trajectory()
    output = SFTTrajectoryCrossEntropyLoss()(
        logits, labels, assistant_mask=assistant, component_ids=components
    )
    valid = assistant[:, 1:]
    per = F.cross_entropy(
        logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), reduction="none"
    ).reshape(1, -1)
    torch.testing.assert_close(output.loss, per[valid].mean())
    assert output.num_valid_tokens.item() == 5


def test_sft_context_and_tool_observations_have_zero_gradient():
    logits, labels, assistant, components = _trajectory()
    output = SFTTrajectoryCrossEntropyLoss()(
        logits, labels, assistant_mask=assistant, component_ids=components
    )
    output.loss.backward()
    # Prediction positions for target system/user/observation tokens.
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, 4]) == 0
    # Tool-call and tool-argument target predictors receive signal.
    assert torch.count_nonzero(logits.grad[:, 2]) > 0
    assert torch.count_nonzero(logits.grad[:, 3]) > 0


def test_sft_component_weights_are_exact():
    logits, labels, assistant, components = _trajectory()
    weights = {int(x): 1.0 for x in SFTComponent}
    weights[int(SFTComponent.FINAL_ANSWER)] = 4.0
    output = SFTTrajectoryCrossEntropyLoss()(
        logits,
        labels,
        assistant_mask=assistant,
        component_ids=components,
        component_weights=weights,
    )
    per = F.cross_entropy(
        logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), reduction="none"
    ).reshape(1, -1)
    aligned_components = components[:, 1:]
    valid = assistant[:, 1:]
    token_weights = torch.ones_like(per)
    token_weights[aligned_components == SFTComponent.FINAL_ANSWER] = 4
    expected = (per * token_weights)[valid].sum() / token_weights[valid].sum()
    torch.testing.assert_close(output.loss, expected)


def test_sft_token_mean_and_sequence_mean_differ_as_documented():
    logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))
    assistant = torch.tensor([[0, 1, 0, 0, 0], [0, 1, 1, 1, 1]], dtype=torch.bool)
    token = SFTTrajectoryCrossEntropyLoss(reduction="token_mean")(
        logits, labels, assistant_mask=assistant
    )
    sequence = SFTTrajectoryCrossEntropyLoss(reduction="sequence_mean")(
        logits, labels, assistant_mask=assistant
    )
    per = F.cross_entropy(
        logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1), reduction="none"
    ).reshape(2, 4)
    expected_sequence = torch.stack(
        (per[0, assistant[0, 1:]].mean(), per[1, assistant[1, 1:]].mean())
    ).mean()
    torch.testing.assert_close(sequence.loss, expected_sequence)
    assert not torch.allclose(token.loss, sequence.loss)


def test_sft_boundary_and_visual_placeholder_masking():
    logits = torch.randn(1, 5, 6)
    labels = torch.tensor([[0, -100, 2, 3, 4]])
    assistant = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.bool)
    boundary = torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.bool)
    output = SFTTrajectoryCrossEntropyLoss()(
        logits, labels, assistant_mask=assistant, boundary_mask=boundary
    )
    assert output.num_valid_tokens.item() == 2


def test_sft_empty_assistant_batch_raises():
    with pytest.raises(ValueError, match="no assistant"):
        SFTTrajectoryCrossEntropyLoss()(
            torch.randn(1, 3, 4),
            torch.tensor([[0, 1, 2]]),
            assistant_mask=torch.zeros(1, 3, dtype=torch.bool),
        )


def test_sft_component_diagnostics_are_detached():
    logits, labels, assistant, components = _trajectory()
    output = SFTTrajectoryCrossEntropyLoss()(
        logits,
        labels,
        assistant_mask=assistant,
        component_ids=components,
        return_per_component=True,
    )
    assert output.per_component_loss_sum
    assert all(not value.requires_grad for value in output.per_component_loss_sum.values())
    assert int(SFTComponent.TOOL_CALL) in output.per_component_num_tokens

