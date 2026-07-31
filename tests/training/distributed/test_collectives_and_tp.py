from __future__ import annotations

import copy
from pathlib import Path

import torch

from src import KimiK3
from src.kimi_k3.config import kimi_k3_cpu_tiny_config
from training.distributed import (
    DataParallelConfig,
    DistributedConfig,
    TensorParallelConfig,
    all_gather_variable,
    build_device_mesh,
    initialize_distributed,
    parallelize_kimi_model,
)
from training.distributed.kimi_tensor_parallel import TensorParallelKDA
from training.optimizer import QKClipController, build_parameter_registry

from tests.training.distributed.helpers import configure_rank, launch


def _tp_worker(rank, world_size, port, output, _unused):
    configure_rank(rank, world_size, port)
    distributed = DistributedConfig(
        enabled=True,
        backend="gloo",
        data_parallel=DataParallelConfig(),
        tensor_parallel=TensorParallelConfig(enabled=True, size=world_size),
    )
    context = build_device_mesh(
        initialize_distributed(distributed), distributed
    )
    try:
        variable = torch.full((rank + 1, 2), float(rank))
        gathered, sizes = all_gather_variable(
            variable, group=context.tp_group
        )
        assert sizes == [1, 2]
        assert gathered.shape == (3, 2)

        torch.manual_seed(17)
        config = kimi_k3_cpu_tiny_config(
            enable_vision=False, enable_mtp=False
        )
        reference = KimiK3(config).eval()
        sharded = copy.deepcopy(reference).eval()
        report = parallelize_kimi_model(sharded, distributed, context)
        assert report["attention_layers_sharded"] == 5
        assert sharded.embed_tokens.weight is sharded.lm_head.weight
        registry = build_parameter_registry(
            sharded, kind="per_head_muon_adamw", strict=True
        )
        assert registry.names_for("per_head_muon")
        assert not registry.missing
        qk_clip = QKClipController(sharded, threshold=100.0)
        assert qk_clip.layers

        input_ids = torch.tensor([[1, 9, 10, 11, 2]])
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        expected = reference(input_ids, attention_mask=mask)
        actual = sharded(input_ids, attention_mask=mask)
        torch.testing.assert_close(
            actual.logits, expected.logits, atol=2e-5, rtol=2e-5
        )

        expected_prefill = reference(
            input_ids[:, :4],
            attention_mask=mask[:, :4],
            use_cache=True,
        )
        actual_prefill = sharded(
            input_ids[:, :4],
            attention_mask=mask[:, :4],
            use_cache=True,
        )
        expected_decode = reference(
            input_ids[:, 4:],
            attention_mask=mask[:, 4:],
            cache=expected_prefill.cache,
            use_cache=True,
        )
        actual_decode = sharded(
            input_ids[:, 4:],
            attention_mask=mask[:, 4:],
            cache=actual_prefill.cache,
            use_cache=True,
        )
        torch.testing.assert_close(
            actual_decode.logits,
            expected_decode.logits,
            atol=3e-5,
            rtol=3e-5,
        )
        for split in range(1, input_ids.shape[1]):
            reference_chunk = reference(
                input_ids[:, :split],
                attention_mask=mask[:, :split],
                use_cache=True,
            )
            parallel_chunk = sharded(
                input_ids[:, :split],
                attention_mask=mask[:, :split],
                use_cache=True,
            )
            reference_pieces = [reference_chunk.logits]
            parallel_pieces = [parallel_chunk.logits]
            for token in range(split, input_ids.shape[1]):
                reference_chunk = reference(
                    input_ids[:, token : token + 1],
                    attention_mask=mask[:, token : token + 1],
                    cache=reference_chunk.cache,
                    use_cache=True,
                )
                parallel_chunk = sharded(
                    input_ids[:, token : token + 1],
                    attention_mask=mask[:, token : token + 1],
                    cache=parallel_chunk.cache,
                    use_cache=True,
                )
                reference_pieces.append(reference_chunk.logits)
                parallel_pieces.append(parallel_chunk.logits)
            torch.testing.assert_close(
                torch.cat(parallel_pieces, dim=1),
                torch.cat(reference_pieces, dim=1),
                atol=3e-5,
                rtol=3e-5,
            )

        padded_ids = torch.tensor(
            [[1, 9, 10, 11, 2], [1, 12, 2, 0, 0]]
        )
        padded_mask = padded_ids.ne(0)
        torch.testing.assert_close(
            sharded(padded_ids, attention_mask=padded_mask).logits,
            reference(padded_ids, attention_mask=padded_mask).logits,
            atol=3e-5,
            rtol=3e-5,
        )
        cloned_cache = actual_prefill.cache.clone()
        reordered_cache = cloned_cache.reorder(torch.tensor([0]))
        assert reordered_cache.sequence_length == 4
        assert torch.equal(
            reordered_cache.sequence_lengths, torch.tensor([4])
        )

        reference.train()
        sharded.train()
        reference.zero_grad()
        sharded.zero_grad()
        expected_loss = reference(
            input_ids, attention_mask=mask, labels=input_ids
        ).loss
        actual_loss = sharded(
            input_ids, attention_mask=mask, labels=input_ids
        ).loss
        expected_loss.backward()
        actual_loss.backward()
        reference_kda = next(
            module
            for module in reference.modules()
            if module.__class__.__name__ == "KimiDeltaAttention"
        )
        sharded_kda = next(
            module
            for module in sharded.modules()
            if isinstance(module, TensorParallelKDA)
        )
        width = sharded_kda.q_proj.local_out_features
        start = rank * width
        torch.testing.assert_close(
            sharded_kda.q_proj.weight.grad,
            reference_kda.projections.q_proj.weight.grad[
                start : start + width
            ],
            atol=1e-3,
            rtol=1e-3,
        )
        torch.optim.SGD(reference.parameters(), lr=0.01).step()
        torch.optim.SGD(sharded.parameters(), lr=0.01).step()
        torch.testing.assert_close(
            sharded_kda.q_proj.weight,
            reference_kda.projections.q_proj.weight[
                start : start + width
            ],
            atol=1e-5,
            rtol=1e-5,
        )
        if rank == 0:
            torch.save(
                {
                    "loss": float(actual_loss),
                    "state_heads": actual_prefill.cache.layer_caches[
                        0
                    ].state.recurrent_state.shape[1],
                },
                output,
            )
    finally:
        context.close()


def test_two_rank_collectives_full_kimi_tp_and_cache_parity(tmp_path):
    output = tmp_path / "tp.pt"
    launch(_tp_worker, output, None)
    result = torch.load(output, weights_only=True)
    assert result["loss"] > 0
    assert result["state_heads"] == 1
