"""Uniform causal-LM batch normalization.

Datasets return already-shifted labels: ``labels[t]`` is the target for
``input_ids[t]``.
"""

import torch

def normalize_lm_batch(batch):
    """
    Convert common batch containers to ``model(**batch)`` format.
    """

    if isinstance(batch, dict):
        if "input_ids" not in batch:
            raise KeyError(
                f"Batch dict must contain 'input_ids'. Available keys: {list(batch.keys())}"
            )

        if "labels" not in batch:
            batch = dict(batch)
            batch["labels"] = batch["input_ids"]

        return batch

    if torch.is_tensor(batch):
        return {
            "input_ids": batch,
            "labels": batch,
        }

    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            input_ids, labels = batch
            return {
                "input_ids": input_ids,
                "labels": labels,
            }

        if len(batch) == 3:
            input_ids, labels, attention_mask = batch
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

        raise ValueError(
            f"Unsupported batch tuple length {len(batch)}; expected 2 or 3."
        )

    raise TypeError(f"Unsupported batch type: {type(batch)}")
