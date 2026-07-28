import json

import torch

from src import CausalLMOutput
from training import JSONLLogger, MemoryLogger, next_token_preview


class PredictableLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, labels=None, attention_mask=None):
        logits = torch.full((*input_ids.shape, 8), -10.0)
        predicted = (input_ids + 1) % 8
        logits.scatter_(-1, predicted.unsqueeze(-1), 10.0)
        return CausalLMOutput(logits=logits, loss=logits.sum() * 0)


def test_next_token_preview_reports_ids_text_and_restores_mode():
    model = PredictableLM().train()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0]]),
        "labels": torch.tensor([[2, 3, 4, -100]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
    }
    preview = next_token_preview(
        model,
        batch,
        max_tokens=2,
        id_to_text=lambda ids: "|".join(map(str, ids)),
    )
    assert preview["input_ids"] == [2, 3]
    assert preview["predicted_ids"] == [3, 4]
    assert preview["next_token_id"] == 4
    assert preview["reference_next_token_id"] == 4
    assert preview["prediction"] == "3|4"
    assert model.training


def test_jsonl_and_memory_loggers_only_store_serializable_scalars(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JSONLLogger(path)
    logger.log(3, {"loss": 1.25, "executed": True})
    logger.close()
    assert json.loads(path.read_text()) == {
        "executed": True,
        "loss": 1.25,
        "step": 3,
    }

    memory = MemoryLogger()
    memory.log(2, {"lr": 1e-3})
    assert memory.records == [{"step": 2, "lr": 1e-3}]
