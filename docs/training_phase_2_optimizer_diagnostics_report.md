# Training Phase 2 — Kimi optimizers and efficient diagnostics

## Scope and source state

- Base commit: `aee68fc` (the Phase 2 worktree is intentionally uncommitted).
- Canonical high-level entry point: `training.train_kimiK3` / `train_kimi_k3`.
- CPU test model: the structurally complete `kimi_k3_cpu_tiny_config`.
- Out of scope remains distributed/sharded Muon, expert-parallel optimizer
  communication, fused CUDA/Triton Newton–Schulz, FP8 states, Hessians,
  Jacobians and per-example gradients.

## Optimizer policy

The default is `per_head_muon_adamw`:

- Q/K/V matrices in KDA, Gated MLA, final Gated MLA and MTP use independent
  Per-Head Muon.
- Other eligible 2D matrices use standard Muon.
- embeddings, norms, biases, router parameters and tied LM-head parameters use
  AdamW with the explicit decay/no-decay split.
- canonical weight decay is `0.1`; Muon inherits it unless separately
  configured.
- Newton–Schulz uses five iterations by default. Shape scaling multiplies a
  semi-orthogonal update by `sqrt(max(rows, cols))`, making its ideal RMS one.
- the hybrid step first validates every gradient. A non-finite gradient skips
  Muon, AdamW, scheduler advancement and QK-Clip as one logical transaction.

The CPU-tiny structural registry contains 50,068 unique trainable parameters:

| Family | tensors | parameters | percentage |
|---|---:|---:|---:|
| AdamW decay | 35 | 2,416 | 4.83% |
| AdamW no decay | 89 | 3,300 | 6.59% |
| Muon | 203 | 40,320 | 80.53% |
| Per-Head Muon | 27 | 4,032 | 8.05% |

Its registry fingerprint is
`6c7e59c595665127ba0a9aacf0519e2354041773917e745df194d30e57d8093e`;
checkpoints store and validate this structural fingerprint.

The 27 Per-Head matrices cover:

- backbone KDA Q/K/V in layers 0–2;
- backbone Gated MLA Q/K/V in layer 3;
- final backbone Gated MLA Q/K/V;
- MTP KDA Q/K/V in layers 0–2;
- MTP Gated MLA Q/K/V in layer 3.

Assignment is structural (owner module type plus layout metadata), not a
substring-only QKV guess.

## Learning-rate schedule and QK-Clip

- AdamW LR default: `3e-4`.
- Muon LR default: equal to AdamW, independently configurable.
- linear warmup: 1% of total steps, with a minimum of one step when applicable.
- the first optimizer update receives warmup step 1; the scheduler advances
  after a successful optimizer transaction only.
- cosine decay supports separate AdamW and Muon minimum LRs.
- Gated MLA QK-Clip uses a detached robust QK-scale proxy
  `RMS(Q) * RMS(K) * sqrt(head_dim)`.
- above threshold, Q and K are each multiplied by
  `sqrt(threshold / observed_scale)`. V and output projections are untouched.
- KDA QK-Clip exists only behind the explicit experimental flag.
- QK counters, consecutive activation state and scheduler state are
  checkpointed.

The exact-resume test runs one hybrid step, serializes both optimizer states,
loads them into an identical model and verifies the next model update and
reported update metrics exactly.

## Diagnostic architecture

Diagnostics have three schedules:

- CHEAP: loss, NTP-only clipped perplexity, throughput, timings, memory,
  global gradients and sampled update health.
- STANDARD: rotating bounded layer samples for block contribution, AttnRes,
  KDA, MLA, MoE, MTP, activation and representation health.
- DEEP: explicit infrequent diagnostic step; it never runs a production-size
  SVD or returns full expert gradients.

All reducers detach before reduction and emit Python floats. Parameter update
monitoring clones only a deterministic contiguous prefix of a bounded number
of tensors. The collector keeps scalar EMA/counter state only. When time or
persistent-memory budgets are exceeded it emits
`DIAGNOSTIC_BUDGET_EXCEEDED`, records degraded mode and temporarily falls back
to CHEAP checks.

Representative healthy output is printed in separate blocks:

```text
Optimizer step 000100
  Convergence
    Total loss / NTP loss / MTP loss / NTP perplexity
  Throughput & runtime
    tokens/s / step / data / forward / backward / optimizer
  Optimization health
    AdamW LR / Muon LR / grad clipping / update ratio / head balance
  Diagnostics budget
    time / fraction / sampled bytes / layer count / degraded mode
  Architecture health
    AttnRes / KDA / MLA / MoE / MTP summaries
  Alerts
    severity / stable code / explanation / observed value
```

The printer uses the same block layout in terminals and Jupyter output.

## Metrics and induced alerts

Each metric family has analytic tests with controlled tensors. Tests compare
the complete emitted dictionary, so adding an untested metric key fails the
test. Important semantic checks include:

- NTP perplexity is `exp(min(loss_ntp, 20))`, never total NTP+MTP loss.
- MTP hidden RMS is elementwise RMS and is distinct from mean vector norm.
- uniform two-source attention entropy normalizes to one.
- residual branch magnitude/state change/cosine match hand calculations.
- MoE load ratios, entropy, QB update RMS and branch utility use controlled
  routing/output values.
- QK-Clip's symmetric analytical factor is checked exactly.
- Per-Head summaries are checked against an explicit loop over heads.
- representation collapse proxies use known feature variances.

Induced pathology tests cover these alert codes:

```text
NONFINITE_METRIC
INACTIVE_BLOCK
ATTNRES_SINGLE_SOURCE_COLLAPSE
ATTNRES_UNIFORM_COLLAPSE
KDA_RETENTION_SATURATION
KDA_STATE_WRITE_CLOSED
KDA_STATE_EXPLOSION
KDA_OUTPUT_GATE_CLOSED
MLA_OUTPUT_GATE_CLOSED
MOE_DEAD_EXPERTS_PERSISTENT
MOE_ROUTED_BRANCH_INACTIVE
MTP_DISCONNECTED
MTP_NO_VALID_TOKENS
QK_CLIP_PERSISTENT
REPRESENTATION_VARIANCE_COLLAPSE
DIAGNOSTIC_BUDGET_EXCEEDED
```

Patience rules require repeated diagnostic observations; NaN/Inf and budget
corruption warnings are immediate.

## Validation and benchmark

- Phase 2 adds 12 focused optimizer/diagnostic test modules plus the real
  Kimi CPU integration test: 63 collected test cases.
- The full repository suite collects 1,363 tests and exits successfully on
  CPU.
- The focused Phase 2 suite exits successfully on CPU (63/63).
- `scripts/benchmark_training_diagnostics.py` provides modes off, cheap,
  standard and deep; reports median/p90 step time, tokens/s, peak GPU memory,
  persistent diagnostic bytes, scalar count and overhead against off.

The checked-in benchmark is a manual measurement tool. CPU one-step smoke
numbers are environment/noise dominated and are recorded only to validate the
script; acceptance overhead must be measured with multiple steps on the target
GPU at contexts 512, 2K and the maximum practical context.

CPU smoke (`context=8`, one step; not an overhead acceptance run):

| mode | median ms | tokens/s | sampled persistent bytes | scalars | overhead |
|---|---:|---:|---:|---:|---:|
| off | 604.62 | 12.97 | 0 | 0 | 0.0% |
| cheap | 620.52 | 12.46 | 50,256 | 13 | 2.63% |
| standard | 707.02 | 10.79 | 50,256 | 308 | 16.94% |
| deep | 690.49 | 11.21 | 50,256 | 308 | 14.20% |

## Project decisions not fixed by Kimi publications

- Muon LR defaults to the AdamW LR rather than an undocumented multiplier.
- QK threshold defaults to `100.0` and is explicitly configurable.
- Newton–Schulz uses the quintic coefficients already established by common
  Muon implementations and five default iterations.
- QK control uses the documented robust RMS proxy rather than claiming KDA has
  global softmax logits.
- diagnostics use deterministic prefix sampling and scalar EMA state.
- no full per-head or per-expert vector is retained during normal logging.

## Remaining limitations

- Real GPU overhead numbers are hardware-dependent and must be filled from the
  manual benchmark before a production-scale run.
- DEEP currently increases diagnostic intent/frequency but intentionally does
  not run recurrent/chunkwise backward parity inside a production training
  step; those audits remain in the architecture test suite.
- QK-Clip is canonical for Gated MLA; KDA clipping remains experimental.
- Distributed, sharded and fused optimizer implementations remain Phase 3+
  work.
