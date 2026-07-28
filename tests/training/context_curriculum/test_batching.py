import pytest
import torch

from training import ProgressiveContextCollator, truncate_batch_to_context


def test_batch_is_truncated_and_all_sequence_fields_remain_aligned():
    batch = {
        "input_ids": torch.arange(20).reshape(2, 10),
        "labels": torch.arange(20).reshape(2, 10) + 1,
        "attention_mask": torch.ones(2, 10, dtype=torch.bool),
        "loss_mask": torch.ones(2, 10, dtype=torch.bool),
        "boundary_mask": torch.ones(2, 10, dtype=torch.bool),
        "token_weights": torch.ones(2, 10),
    }
    truncated, metrics = truncate_batch_to_context(batch, 6)
    assert all(
        truncated[name].shape[:2] == (2, 6)
        for name in (
            "input_ids", "labels", "attention_mask", "loss_mask",
            "boundary_mask", "token_weights",
        )
    )
    assert metrics == {
        "valid_tokens": 12.0,
        "padding_fraction": 0.0,
        "sequence_length": 6.0,
    }


def test_dynamic_padding_stops_at_longest_item_not_final_context():
    collator = ProgressiveContextCollator(
        16, pad_token_id=0, ignore_index=-100
    )
    batch = collator([
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5, 6, 7, 8]),
         "labels": torch.tensor([4, 5, 6, 7, 8])},
    ])
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    assert batch["labels"][0, 3:].tolist() == [-100, -100]


def test_existing_padding_is_trimmed_to_batch_need_and_measured():
    batch = {
        "input_ids": torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]]),
        "labels": torch.tensor([[1, 2, -100, -100], [3, 4, 5, -100]]),
        "attention_mask": torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        ),
    }
    result, metrics = truncate_batch_to_context(batch, 8)
    assert result["input_ids"].shape == (2, 3)
    assert metrics["padding_fraction"] == pytest.approx(1 / 6)
    assert metrics["valid_tokens"] == 5


def test_packed_segments_are_rejected_instead_of_leaking_between_documents():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "segment_ids": torch.tensor([[0, 0, 1, 1]]),
    }
    with pytest.raises(ValueError, match="segment-isolated"):
        truncate_batch_to_context(batch, 4)


def test_single_segment_ids_are_preserved_and_boundary_mask_is_aligned():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0]]),
        "labels": torch.tensor([[1, 2, 3, -100]]),
        "attention_mask": torch.tensor([[True, True, True, False]]),
        "segment_ids": torch.tensor([[7, 7, 7, -1]]),
    }
    result, _ = truncate_batch_to_context(batch, 4)
    assert result["segment_ids"].shape == result["input_ids"].shape
    assert result["boundary_mask"].tolist() == [[True, True, True]]


def test_multimodal_truncation_never_silently_detaches_visual_inputs():
    batch = {
        "input_ids": torch.tensor([[5, 5, 3, 3]]),
        "labels": torch.tensor([[5, 5, 3, 3]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "pixel_values": torch.randn(1, 3, 16, 16),
    }
    with pytest.raises(ValueError, match="detach pixel_values"):
        truncate_batch_to_context(batch, 2, image_token_id=3)
    safe, _ = truncate_batch_to_context(batch, 4, image_token_id=3)
    assert safe["pixel_values"] is batch["pixel_values"]


def test_invalid_alignment_and_all_padding_fail_before_model_forward():
    with pytest.raises(ValueError, match="align"):
        truncate_batch_to_context(
            {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "labels": torch.ones(1, 3, dtype=torch.long),
            },
            4,
        )
    with pytest.raises(ValueError, match="all-padding"):
        truncate_batch_to_context(
            {
                "input_ids": torch.zeros(1, 4, dtype=torch.long),
                "attention_mask": torch.zeros(1, 4, dtype=torch.bool),
            },
            4,
        )


def test_non_right_padded_mask_fails_instead_of_dropping_valid_tail():
    with pytest.raises(ValueError, match="right-padded"):
        truncate_batch_to_context(
            {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.tensor([[True, False, True, False]]),
            },
            4,
        )
