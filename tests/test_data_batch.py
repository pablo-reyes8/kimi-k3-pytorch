import pytest
import torch

from data.batch import normalize_lm_batch


def tensors():
    return torch.arange(8).view(2, 4), torch.arange(8, 16).view(2, 4)


def test_dict_with_labels_is_returned_without_copy():
    input_ids, labels = tensors()
    batch = {"input_ids": input_ids, "labels": labels, "metadata": "kept"}
    normalized = normalize_lm_batch(batch)
    assert normalized is batch
    assert normalized["metadata"] == "kept"


def test_dict_without_labels_is_copied_and_uses_same_input_tensor_as_labels():
    input_ids, _ = tensors()
    original = {"input_ids": input_ids}
    normalized = normalize_lm_batch(original)
    assert normalized is not original
    assert "labels" not in original
    assert normalized["labels"] is input_ids


def test_tensor_batch_maps_to_input_and_labels_without_copying():
    input_ids, _ = tensors()
    normalized = normalize_lm_batch(input_ids)
    assert normalized["input_ids"] is input_ids
    assert normalized["labels"] is input_ids


def test_two_and_three_item_sequences_map_to_documented_fields():
    input_ids, labels = tensors()
    mask = torch.ones_like(input_ids)
    pair = normalize_lm_batch((input_ids, labels))
    triple = normalize_lm_batch([input_ids, labels, mask])
    assert pair == {"input_ids": input_ids, "labels": labels}
    assert triple == {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": mask,
    }


@pytest.mark.parametrize("batch", [(), (1,), (1, 2, 3, 4)])
def test_unsupported_sequence_lengths_rejected(batch):
    with pytest.raises(ValueError, match="expected 2 or 3"):
        normalize_lm_batch(batch)


def test_dict_missing_input_ids_rejected_with_available_keys():
    with pytest.raises(KeyError, match="Available keys"):
        normalize_lm_batch({"labels": torch.ones(2, 3)})


@pytest.mark.parametrize("batch", [None, 3.14, "tokens", {"a", "b"}])
def test_unsupported_types_rejected(batch):
    with pytest.raises(TypeError):
        normalize_lm_batch(batch)
