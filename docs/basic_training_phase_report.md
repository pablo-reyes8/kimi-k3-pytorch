# Kimi K3 Mini — Basic Training Engine Report

## Scope

This phase adapts the existing DeepSeek-style training layout to Kimi K3
without moving training logic into a monolithic CLI. The public high-level
entry point is:

```python
from training import train_kimiK3
```

It orchestrates the already separate seed, precision, AdamW, scheduler, EMA,
epoch loops, evaluation, qualitative prediction, logging and checkpoint
modules.

The starting repository revision was `1119dd2`.

## Source of each decision

| Source | Preserved in this phase |
|---|---|
| Kimi K3 report | joint NTP/MTP objective, Quantile Balancing, cosine schedule and 1% warmup default |
| Existing project contracts | `KimiK3.forward`, typed NTP/MTP loss outputs, NoPE inputs, MoE routing-bias buffers |
| Kimi-K3 Mini engineering | AdamW baseline, token-exact accumulation, global gradient clipping, EMA option, JSONL logs, epoch checkpoints and CPU-sized tests |

Muon, Per-Head Muon, Newton–Schulz, Kimi weight clipping, distributed
training, SFT, RL and full multimodal data mixing remain outside this phase.

## Public API and file layout

- `training/config.py`: validated immutable configuration.
- `training/state.py`: serializable counters.
- `training/train_one_epoch.py`: one complete training epoch.
- `training/eval_one_epoch.py`: side-effect-free token-weighted evaluation.
- `training/train_kimi_k3.py`: `train_kimiK3` orchestration.
- `training/loss_accounting.py`: typed NTP/MTP normalization.
- `training/moe_control.py`: logical Quantile Balancing windows.
- `training/predictions.py`: periodic next-token inspection.
- `training/logger.py`: JSONL and in-memory logging.
- `training/curriculum.py`: context-stage selection and validation.
- `training/checkpoints.py`: atomic checkpoint/resume.

Existing optimizer, scheduler, AMP, EMA and seed modules remain the canonical
implementations and are reused by the orchestrator.

## Accumulation equation

Kimi NTP and MTP use independent normalizers:

\[
L_{\text{window}}
=
\frac{\sum_m L^{NTP}_{sum,m}}{\sum_m N^{NTP}_m}
+
\lambda_{MTP}
\frac{\sum_m L^{MTP}_{sum,m}}{\sum_m N^{MTP}_m}.
\]

The basic reference engine retains the microbatch graphs until a logical
window closes and performs one backward pass on this exact expression. This
uses more memory than a distributed production loop, but it avoids weighting
microbatches equally when they contain different amounts of padding and
avoids incorrectly sharing one normalizer between NTP and MTP.

Generic causal LMs that expose only a scalar loss use a valid-token-weighted
compatibility fallback.

## Update order

For each logical optimizer window:

1. zero gradients and open Quantile Balancing accumulation;
2. run all forwards under the selected autocast policy;
3. combine differentiable loss sums using the equation above;
4. backward;
5. unscale FP16 gradients;
6. reject non-finite loss or gradients;
7. record the pre-clip global norm;
8. apply global gradient clipping;
9. run the optimizer step;
10. on a successful step, advance scheduler, EMA and Quantile Balancing;
11. on an AMP overflow, discard pending Quantile Balancing state;
12. clear gradients.

The scheduler therefore advances by optimizer updates, never microbatches.

## Quantile Balancing lifecycle

Every `StableLatentMoE` layer keeps the same routing bias throughout all
microbatches in a logical batch. Exact backends retain detached score/cutoff
statistics; histogram backends accumulate bounded histograms. The controller
commits all next biases only after an optimizer step succeeds. Exceptions and
skipped AMP updates discard pending state without changing the active bias.

Evaluation does not open a balancing window and cannot mutate routing bias.
Routing biases remain ordinary model buffers and are included in
`model.state_dict()`.

## Evaluation and qualitative prediction

Evaluation aggregates loss sums by valid tokens and reports NTP perplexity
separately from the mixed NTP+MTP objective. It restores the model's original
training mode and supports a reversible EMA swap.

At the configured epoch frequency, `next_token_preview` records:

- the inspected context token IDs/text;
- teacher-forced predicted IDs;
- reference IDs;
- the final predicted and reference next-token IDs.

No tokenizer dependency is required; raw IDs are used as the fallback.

## Checkpoint format

Format version 2 stores:

- model, optimizer, scheduler and scaler states;
- optional EMA state;
- `TrainerState`;
- optional context curriculum state;
- epoch, global optimizer step and history;
- Python, NumPy, Torch CPU and Torch CUDA RNG states;
- model/training configuration and metadata.

Writes use a sibling temporary file followed by `os.replace`, so a valid
checkpoint is never overwritten partially. Model restore is strict by
default.

Epoch-boundary resume restores the states above. Exact continuation still
depends on the supplied dataloader/sampler reproducing its order; arbitrary
mid-epoch sampler cursor serialization is intentionally not claimed here.

## Tests

Seventeen new test functions, with parameterized cases, cover:

- config and state validation;
- curriculum boundaries and serialization;
- variable-padding accumulation versus a concatenated large batch;
- partial accumulation windows and counters;
- evaluation weighting, mode restoration and no mutation;
- qualitative next-token preview;
- JSONL logging;
- exact and histogram Quantile Balancing timing and discard;
- real Kimi tiny NTP+MTP gradient flow;
- real Kimi tiny per-layer Quantile Balancing commits;
- high-level train/eval/preview/checkpoint integration;
- version-2 checkpoint restoration;
- exact CPU continuation versus a checkpoint-split training trajectory.

The complete repository test suite passes on CPU. CUDA-only and optional
external-oracle cases retain their existing explicit skips.

## Current limitations

- The exact reference accumulation strategy retains graphs for one logical
  window; a memory-optimized distributed reduction can be added later without
  changing the public loop.
- Checkpoints are emitted at clean epoch/optimizer boundaries, not arbitrary
  microbatch boundaries.
- Dataloader/sampler cursor state is not serialized by the generic engine.
- Advanced MoE/KDA diagnostic scheduling and the Kimi-specific optimizer work
  remain follow-up phases.
