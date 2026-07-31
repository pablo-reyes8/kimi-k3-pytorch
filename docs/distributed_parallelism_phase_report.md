# Distributed parallelism phase report

## Outcome

Kimi-K3 Mini now has a PyTorch-native distributed baseline integrated into the
existing `train_kimiK3` orchestrator. The implementation preserves the single
model definition and transforms its owned modules in place before optimizer
construction.

## Topology and lifecycle

The strict training YAML schema defines independent DP, TP and EP dimensions:

```text
world_size = DP × TP × EP
rank = (dp_rank × TP + tp_rank) × EP + ep_rank
```

Every process constructs the same groups in deterministic order. Environment
parsing validates `RANK`, `WORLD_SIZE`, `LOCAL_RANK` and
`LOCAL_WORLD_SIZE`; `env://` initialization is owned and destroyed exactly
once by the entry point. NCCL is selected for CUDA and Gloo for CPU when
`backend: auto`.

The data sampler shards over DP×EP while keeping TP peers on identical token
batches. It serializes both epoch and cursor. Valid-token losses, sample/token
counters and PCC thresholds are reduced globally without counting TP replicas.

## Data parallelism

DDP is scoped to the DP process group and uses PyTorch gradient buckets.
The trainer already performs one backward per completed accumulation window,
so it synchronizes exactly once per optimizer transaction. Non-finite state is
checked across all ranks before the optimizer transition.

FSDP is applied before optimizer construction, supports PyTorch
`FULL_SHARD`, `SHARD_GRAD_OP` and `NO_SHARD` policies, and uses Kimi attention
layers as its structural wrapping boundary. EMA+FSDP is rejected until a
sharded EMA exists. On PyTorch builds where FSDP requires an accelerator, the
CPU path raises a precise error and the CPU suite tests DDP instead.

## Kimi tensor parallelism

Tensor parallelism owns complete attention heads:

- KDA Q/K/V, beta, decay, head RMSNorm, recurrent matrix and ShortConv state
  are sliced on head boundaries;
- MLA Q/K/V reconstruction heads are sliced while the compressed latent cache
  remains replicated;
- full-rank gates are column-sharded and output projections are row-sharded;
- the tied token embedding and LM head share one vocabulary-local parameter;
- gathered logits preserve the existing public `KimiK3Output` and inference
  APIs;
- Per-Head Muon registry metadata describes local complete heads.

Attention outputs use differentiable collectives. Prefill and token decode
return the same logical outputs as the unsharded model, and cache tests verify
local KDA head ownership plus replicated MLA latents.

Stable LatentMoE, AttnRes, norms and vision remain replicated on the TP axis in
this baseline. `shard_moe_latent_projections: true` is rejected rather than
silently ignored.

## Expert parallelism

Routed experts use contiguous ownership and no capacity drop. Each rank:

1. computes the replicated router and latent projection;
2. sorts assignments by expert owner;
3. exchanges variable split sizes;
4. dispatches latent tokens and unbiased routing weights with autograd-aware
   `all_to_all_single`;
5. evaluates only locally owned experts;
6. returns weighted outputs to their source ranks and performs sparse combine.

Shared experts remain replicated. Exact Quantile Balancing gathers score
windows across EP; histogram QB sums its counts before the next bias is
committed. Global expert loads feed diagnostics.

This is a canonical no-drop PyTorch baseline, not MoonEP's redundant-expert
planner, static-shape scheduler or fused communication kernels.

## Checkpointing

Distributed checkpoints are atomic directories:

```text
run_epoch_0000/
├── metadata.json
├── rank_00000.pt
├── rank_00001.pt
└── SUCCESS
```

`SUCCESS` is written last, followed by an atomic directory rename. Each shard
contains model, optimizer, scheduler, scaler, optional EMA, trainer/PCC/
diagnostic state, sampler cursor and per-rank Python/NumPy/Torch RNG. Metadata
records topology and configuration fingerprints. Resume rejects topology
changes explicitly.

## Launch profiles

```bash
python -m scripts.validate_distributed_config \
  --profile config/kimi_full_pipeline/distributed_tp_ep_4x_24gb

torchrun --standalone --nproc_per_node=4 \
  -m scripts.train_kimi \
  --profile config/kimi_full_pipeline/distributed_tp_ep_4x_24gb
```

No training or dataset download is performed by the validation command.

## Verification

CPU/Gloo tests exercise real collectives and numerical behavior:

- environment, mesh coordinates, strict YAML and sampler resume;
- two-rank DDP versus one global-batch update;
- master-orchestrator DDP smoke and atomic checkpoint roundtrip;
- two-rank full Kimi TP forward, backward, prefill and decode parity;
- two-rank no-drop EP output, input/expert gradient and global-QB parity;
- four-rank TP×EP full-model forward;
- world-size-one row/column/vocabulary primitives and CE;
- explicit CPU FSDP capability boundary.

CUDA/NCCL performance measurements remain future empirical work.
