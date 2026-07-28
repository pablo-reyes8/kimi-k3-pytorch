import torch
from torch.utils.data import DataLoader

from data.inspection import (
    decode_preview,
    inspect_lm_dataloader,
    summarize_lm_batch,
    summarize_tensor,
)
from data.synthetic_long_context_retrieval import SimpleWordTokenizer


def test_summarize_tensor_reports_exact_metadata_and_statistics():
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    summary = summarize_tensor(tensor)
    assert summary == {
        "shape": [2, 2],
        "dtype": "int64",
        "device": "cpu",
        "numel": 4,
        "min": 1.0,
        "max": 4.0,
        "mean": 2.5,
    }


def test_summarize_empty_tensor_omits_reductions():
    summary = summarize_tensor(torch.empty(0, 3))
    assert summary["numel"] == 0
    assert "min" not in summary and "mean" not in summary


def test_summarize_batch_normalizes_and_ignores_metadata():
    ids = torch.ones(2, 3, dtype=torch.long)
    summary = summarize_lm_batch({"input_ids": ids, "source": "synthetic"})
    assert set(summary) == {"input_ids", "labels"}
    assert summary["input_ids"]["shape"] == [2, 3]


def test_decode_preview_uses_tokenizer_then_mapping_then_numeric_fallback():
    tokenizer = SimpleWordTokenizer()
    tokenizer.add_token("hello")
    ids = torch.tensor([tokenizer.token_to_idx["hello"], 99])
    assert decode_preview(ids, tokenizer, max_tokens=1) == "hello"

    class MappingOnly:
        idx_to_token = {1: "one", 2: "two"}

    assert decode_preview(torch.tensor([1, 2]), MappingOnly()) == "one two"
    assert decode_preview(torch.tensor([1, 2]), object()) == "1 2"


def test_dataloader_inspection_limits_batches_and_adds_previews():
    tokenizer = SimpleWordTokenizer()
    tokenizer.add_token("hello")
    token = tokenizer.token_to_idx["hello"]
    samples = [
        {"input_ids": torch.tensor([token, token]), "labels": torch.tensor([token, token])}
        for _ in range(4)
    ]
    result = inspect_lm_dataloader(
        DataLoader(samples, batch_size=1), tokenizer=tokenizer, num_batches=2
    )
    assert result["num_batches_inspected"] == 2
    assert len(result["batches"]) == 2
    assert result["batches"][0]["input_preview"] == "hello hello"
