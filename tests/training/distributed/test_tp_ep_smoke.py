from __future__ import annotations

import copy

import torch

from src import KimiK3
from src.kimi_k3.config import kimi_k3_cpu_tiny_config
from training.distributed import (
    DataParallelConfig,
    DistributedConfig,
    ExpertParallelConfig,
    TensorParallelConfig,
    build_device_mesh,
    initialize_distributed,
    parallelize_kimi_model,
)

from tests.training.distributed.helpers import configure_rank, launch


def _tp_ep_worker(rank, world_size, port, output, _unused):
    configure_rank(rank, world_size, port)
    distributed = DistributedConfig(
        enabled=True,
        backend="gloo",
        data_parallel=DataParallelConfig(),
        tensor_parallel=TensorParallelConfig(enabled=True, size=2),
        expert_parallel=ExpertParallelConfig(enabled=True, size=2),
    )
    context = build_device_mesh(
        initialize_distributed(distributed), distributed
    )
    try:
        torch.manual_seed(71)
        config = kimi_k3_cpu_tiny_config(
            enable_vision=False, enable_mtp=False
        )
        full = KimiK3(config).eval()
        reference = copy.deepcopy(full).eval()
        report = parallelize_kimi_model(full, distributed, context)
        ids = (
            torch.tensor([[1, 9, 10, 2]])
            if context.ep_rank == 0
            else torch.tensor([[1, 11, 12, 2]])
        )
        mask = torch.ones_like(ids, dtype=torch.bool)
        expected = reference(ids, attention_mask=mask).logits
        actual = full(ids, attention_mask=mask).logits
        torch.testing.assert_close(
            actual, expected, atol=3e-5, rtol=3e-5
        )
        expected_prefill = reference(
            ids[:, :3],
            attention_mask=mask[:, :3],
            use_cache=True,
        )
        actual_prefill = full(
            ids[:, :3],
            attention_mask=mask[:, :3],
            use_cache=True,
        )
        expected_decode = reference(
            ids[:, 3:],
            attention_mask=mask[:, 3:],
            cache=expected_prefill.cache,
            use_cache=True,
        )
        actual_decode = full(
            ids[:, 3:],
            attention_mask=mask[:, 3:],
            cache=actual_prefill.cache,
            use_cache=True,
        )
        torch.testing.assert_close(
            actual_decode.logits,
            expected_decode.logits,
            atol=3e-5,
            rtol=3e-5,
        )
        full.train()
        loss = full(ids, attention_mask=mask, labels=ids).loss
        loss.backward()
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in full.parameters()
            if parameter.grad is not None
        )
        torch.optim.SGD(full.parameters(), lr=0.01).step()
        assert report["tensor_parallel"]
        assert report["expert_parallel"]
        assert report["moe_layers_sharded"] == 5
        if rank == 0:
            torch.save(
                {
                    "world_size": context.world_size,
                    "attention_layers": report[
                        "attention_layers_sharded"
                    ],
                },
                output,
            )
    finally:
        context.close()


def test_four_rank_tp_ep_full_model_smoke(tmp_path):
    output = tmp_path / "tp_ep.pt"
    launch(_tp_ep_worker, output, None, world_size=4)
    result = torch.load(output, weights_only=True)
    assert result == {"world_size": 4, "attention_layers": 5}
