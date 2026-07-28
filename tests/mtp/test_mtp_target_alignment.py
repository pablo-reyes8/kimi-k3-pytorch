import pytest
import torch

from src.mtp import build_mtp_training_view


def test_mtp_alignment_exact_for_length_five():
    hidden = torch.arange(5.0).view(1, 5, 1)
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    view = build_mtp_training_view(hidden, ids)
    torch.testing.assert_close(view.source_hidden, hidden[:, :3])
    assert view.future_input_ids.tolist() == [[11, 12, 13]]
    assert view.target_ids.tolist() == [[12, 13, 14]]
    assert view.valid_mask.tolist() == [[True, True, True]]


@pytest.mark.parametrize("tokens,expected", [(0, 0), (1, 0), (2, 0), (3, 1)])
def test_short_sequences_have_exact_mtp_length(tokens, expected):
    hidden = torch.randn(2, tokens, 4)
    ids = torch.zeros(2, tokens, dtype=torch.long)
    view = build_mtp_training_view(hidden, ids)
    assert view.source_hidden.shape == (2, expected, 4)
    assert view.future_input_ids.shape == (2, expected)
    assert view.target_ids.shape == (2, expected)
    assert view.valid_mask.shape == (2, expected)


def test_external_labels_are_shifted_once_only():
    hidden = torch.randn(1, 5, 3)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = torch.tensor([[101, 102, 103, 104, 105]])
    view = build_mtp_training_view(hidden, ids, labels=labels)
    assert view.future_input_ids.tolist() == [[2, 3, 4]]
    assert view.target_ids.tolist() == [[103, 104, 105]]


def test_ignore_index_only_removes_corresponding_target():
    hidden = torch.randn(1, 6, 3)
    ids = torch.arange(6).view(1, 6)
    labels = ids.clone()
    labels[0, 3] = -100
    view = build_mtp_training_view(hidden, ids, labels=labels)
    assert view.valid_mask.tolist() == [[True, False, True, True]]


@pytest.mark.parametrize(
    "mask,expected",
    [
        ([1, 1, 1, 1, 0, 0], [1, 1, 0, 0]),
        ([0, 0, 1, 1, 1, 1], [0, 0, 1, 1]),
        ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0]),
    ],
)
def test_padding_uses_local_three_position_rule(mask, expected):
    hidden = torch.randn(1, 6, 3)
    ids = torch.arange(6).view(1, 6)
    view = build_mtp_training_view(
        hidden, ids, attention_mask=torch.tensor([mask], dtype=torch.bool)
    )
    assert view.valid_mask.tolist() == [
        [bool(value) for value in expected]
    ]


def test_segment_boundaries_never_form_cross_document_targets():
    hidden = torch.randn(1, 7, 3)
    ids = torch.arange(7).view(1, 7)
    segments = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    view = build_mtp_training_view(hidden, ids, segment_ids=segments)
    assert view.valid_mask.tolist() == [[True, False, False, True, True]]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("input_ids", torch.zeros(2, 4), TypeError),
        ("attention_mask", torch.ones(2, 4), TypeError),
        ("labels", torch.zeros(2, 4), TypeError),
        ("segment_ids", torch.zeros(2, 4), TypeError),
    ],
)
def test_alignment_rejects_invalid_dtypes(field, value, error):
    kwargs = dict(
        last_hidden_state=torch.randn(2, 4, 3),
        input_ids=torch.zeros(2, 4, dtype=torch.long),
    )
    kwargs[field] = value
    with pytest.raises(error):
        build_mtp_training_view(**kwargs)


def test_alignment_view_is_contiguous_slice_semantics():
    hidden = torch.randn(2, 8, 4)
    ids = torch.randint(0, 10, (2, 8))
    view = build_mtp_training_view(hidden, ids)
    assert view.source_hidden.storage_offset() == hidden[:, :-2].storage_offset()
    assert view.future_input_ids.storage_offset() == ids[:, 1:-1].storage_offset()
