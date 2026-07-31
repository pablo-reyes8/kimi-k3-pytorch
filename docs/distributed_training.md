# Distributed training

Distributed execution remains part of the ordinary three-YAML pipeline. The
model is built once, transformed in place, and passed to the same
`train_kimiK3` orchestrator used by single-device runs.

## Validate before launch

Validation parses all three files and checks the DP×TP×EP product, KDA/MLA
head ownership, vocabulary shards, routed-expert ownership, sparse-DDP
requirements and unsupported combinations. It does not allocate a model,
build a dataset or initialize a process group.

```bash
python -m scripts.validate_distributed_config \
  --profile config/kimi_full_pipeline/distributed_tp_ep_4x_24gb
```

The command prints the matching `torchrun` invocation. For example:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m scripts.train_kimi \
  --profile config/kimi_full_pipeline/distributed_tp_ep_4x_24gb
```

`WORLD_SIZE` must equal `DP × TP × EP`. `backend: auto` selects NCCL on CUDA
and Gloo on CPU. `LOCAL_RANK` owns CUDA device selection.

## Supported axes

| Axis | Implemented ownership |
|---|---|
| DP | DDP or FSDP on the DP process group; data is sharded |
| TP | Complete KDA/MLA heads and tied vocabulary ranges |
| EP | Contiguous routed experts with no-drop variable all-to-all |

TP peers consume identical batches. Data is sharded over DP×EP, so EP can
route distinct local tokens while TP peers remain aligned. Loss accounting,
validation and PCC token thresholds use globally reduced valid-token counts
without counting TP replicas.

DDP over sparse routed experts requires `find_unused_parameters: true`. EMA
with FSDP is rejected before allocation because a sharded EMA is not present.
PP and cross-device KDA context parallelism are reserved interfaces and must
remain disabled.

## Checkpoints

Distributed checkpoints are atomic directories containing one shard per
rank, metadata and a final `SUCCESS` marker. Resume supports only the same
DP/TP/EP topology. Model, optimizer, scheduler, scaler, trainer, PCC,
diagnostic, sampler and per-rank RNG state are restored together.

Use only a fresh output directory or a valid `resume_from` path. A directory
without `SUCCESS` is deliberately rejected.

