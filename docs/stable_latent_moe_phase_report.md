# Kimi K3 Stable LatentMoE phase report

## Scope and provenance

This phase implements the channel-mixing axis of the textual architecture:

```text
sequence mixing: KDA + Gated MLA
layer mixing:    Full/Block Attention Residuals
channel mixing:  Stable LatentMoE
```

The Kimi K3 equations and phase specification are the source of truth. The
older DeepSeek MoE implementation was inspected only for reusable engineering
ideas—independent `ModuleList` experts and expert-grouped dispatch. Its
sqrt-softplus/hash routing, full-width routed branch, branch scales and
auxiliary balancing losses were deliberately not carried into Kimi.

The starting phase-7 worktree at HEAD `17c6800` passed 1,186 tests with
17 skips in 29.52 seconds.

## Public API and files

The package `src/stable_latent_moe/` exposes:

```text
StableLatentMoEConfig
SharedExpert
RoutedExpert
TopKRouter
reference_sparse_dispatch
vectorized_sparse_dispatch
ExactQuantileBalancer
HistogramQuantileBalancer
StableLatentMoE
RouterOutput
StableLatentMoEOutput
MoEDiagnostics
QuantileBalanceUpdate
```

`src/kimi_block.py` adds `KimiBlockConfig` and `KimiBlock`. Configuration
examples live in:

```text
config/model/kimi_block_tiny.yaml
config/model/kimi_k3_architecture.yaml
```

## Canonical forward and shapes

For `M=B*T`, full width `D`, latent width `L`, routed experts `E` and Top-k
`k`:

```text
x_flat                         [M,D]
shared experts                 N_s × ([M,D] -> [M,D])
z = W_down(x_flat)             [M,L]
sigmoid(W_router(x_flat))      [M,E]
Top-k(raw scores + bias)       [M,k]
unbiased normalized weights    [M,k]
sparse expert aggregate        [M,L]
RMSNorm(aggregate)             [M,L]
W_up(normalized aggregate)     [M,D]
shared sum + routed output     [B,T,D]
```

Shared experts are full-width, always active, independent and summed without
averaging. Routed experts operate exclusively in latent width. `W_down` and
`W_up` each run once per token, outside individual experts. RMSNorm is placed
after weighted aggregation and before `W_up`. The result contains no input
residual.

## Router and Quantile Balancing

The router uses sigmoid affinities. Dispatch uses `raw_score + routing_bias`,
while mixture weights use only the selected raw scores divided by their raw
sum. The routing bias is a persistent buffer, never an optimizer parameter,
and ties follow documented `torch.topk` behavior on the current device.

Exact QB uses the biased Top-(k+1) cutoff, margins `raw_score - cutoff`, the
`1-k/E` quantile and a mean-centered next bias. Compute and commit are
separate and execute under `no_grad`. The current forward is completed with
the old bias; only a future forward sees the update.

The histogram backend stores only integer `[E,num_bins]` counts plus explicit
underflow/overflow counters. It supports resettable logical-batch
microbatch accumulation and linear interpolation between the histogram bins
containing the lower and upper quantile ranks. Evaluation and decode freeze
the bias and never accumulate histograms.

## Backends and numerical results

The reference backend loops over token assignments and is the mathematical
oracle. The vectorized backend flattens assignments, sorts/groups by expert,
executes only active experts and scatters weighted results back to their
original tokens.

Fixed-seed FP64 observations:

```text
reference vs vectorized max abs:          6.78e-21
reference vs vectorized gradient max abs: 1.06e-22
exact vs 256-bin histogram max bias diff: 2.58e-3
256-bin width:                            7.81e-3
KimiBlock full vs prefill max abs:        0.0
KimiBlock full vs token decode max abs:   1.11e-16
```

CPU BF16 versus FP32 on the tiny MoE produced maximum/mean absolute
differences of `2.75e-6` and `4.83e-7`; outputs and gradients remained finite.
CUDA FP32/BF16/FP16 tests are included and skip explicitly when CUDA is
unavailable.

## Parameters, state and diagnostics

Without biases:

```text
P = N_s(3 D H_s) + E(3 L H_r) + 2DL + ED + L
```

For the tiny fixture `D=8,L=4,N_s=2,E=4,k=2,H_s=6,H_r=5`, expected and
observed parameters are both 628. The routing bias is included in
`state_dict` but excluded from parameter count. A one-pattern `KimiBlock`
contains five independent MoE instances, 11 AttnRes sites and 4,906 total
parameters.

Diagnostics report assignment/load distributions, coefficient of variation,
zero-load experts, selected-weight range and entropy, bias statistics, shared
and routed RMS values, and routed aggregate RMS before/after normalization.
Normal forwards retain no `[M,E]` score or margin matrices.

MoE introduces no temporal cache. `HybridBackboneCache` remains exclusively
the ordered KDA/MLA state. Histogram state exists only during an explicit
training accumulation window.

## Integration and KimiBlock

Every KDA, periodic Gated MLA and final global Gated MLA owns an independent
Stable LatentMoE. AttnRes still contributes exactly two sites per transformer
layer: pre-attention and pre-channel-mixer. Only the total MoE output is
recorded in depth history.

`KimiBlock` defaults to the canonical 3 KDA : 1 MLA repeated pattern but makes
the explicit sequence configurable for reduced research ablations. It always
retains the final global Gated MLA and attaches a MoE to every transformer
layer. The full metadata profile represents 69 KDA, 24 MLA and 93 independent
MoE layers without allocating the 896-expert model in tests.

This is not the final model class: `KimiBlock` accepts hidden states and
returns the typed backbone output. Token embeddings, LM heads, language-model
loss, generation and the single final Kimi K3 orchestrator were intentionally
not implemented.

## Tests and benchmark

Phase 8 adds 85 tests: 82 CPU passes and three CUDA skips. They cover config,
closed-form parameter counts, manual expert/router equations, biased Top-k,
unbiased weights, reference/vectorized dispatch and gradients, zero-load
experts, exact and histogram QB, controlled 8-token balancing, microbatch
accumulation, branch isolation, norm placement, gradcheck, extreme numerics,
BF16, serialization, token/batch isolation, configurable KimiBlock topology,
AttnRes history, diagnostics and full/prefill/decode.

Final repository regression:

```text
1,288 collected
1,268 passed
20 skipped (CUDA or optional oracle)
0 failed
30.34 seconds on CPU
```

`benchmarks/benchmark_stable_latent_moe.py` measures reference/vectorized
forward and backward for natural and deliberately imbalanced routes, `k=1`
and `k>1`, including many zero-load experts. In a one-repeat CPU run with
`D=32,L=16,E=8,M=256,k=2`:

```text
reference natural:     73.23 ms forward, 195.99 ms forward+backward
vectorized natural:     2.07 ms forward,   5.02 ms forward+backward
reference imbalanced:  71.49 ms forward, 155.71 ms forward+backward
vectorized imbalanced:  1.52 ms forward,   3.46 ms forward+backward
```

These are local pure-PyTorch measurements, not comparisons with MoonEP or
claims about production distributed performance.

## Known limits

The phase intentionally excludes expert parallelism, all-to-all, capacity or
token dropping, expert replication, distributed histogram reduction, fused
CUDA/Triton kernels and low-precision production formats. No auxiliary
balancing loss is present. Cross-device ordering under exact Top-k ties is not
promised.

The next phase should build the final orchestrator around these stable blocks;
it should not redesign KDA, Gated MLA, AttnRes or Stable LatentMoE.
