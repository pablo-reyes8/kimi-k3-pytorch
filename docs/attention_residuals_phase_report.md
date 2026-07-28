# Kimi K3 Attention Residuals phase report

## 1. Files created and modified

The implementation lives in `src/attention_residuals/`: configuration,
metadata, typed outputs, the canonical depth-softmax site, Full and Block
states/controllers, online-softmax utilities, two-phase preprocessing and
diagnostics. Integration changes are isolated to `src/hybrid_backbone/` and
the public exports in `src/__init__.py`.

Tests are grouped in `tests/attention_residuals/`. This phase also adds
`benchmarks/benchmark_attention_residuals.py`, updates `docs/testing.md`, and
corrects the canonical final-FFN accounting in the phase-6 report.

## 2. Phase-6 architecture audit

The canonical pattern remains exactly:

```text
KDA -> KDA -> KDA -> Gated MLA
```

There is one additional final Gated MLA. For `G` groups the implementation
therefore exposes `3G` KDA layers, `G+1` MLA layers and `4G+1` complete
transformer layers. `G=23` represents the reported 69 KDA + 24 MLA = 93-layer
topology without instantiating it during CPU tests.

The recorded pre-phase baseline at commit `45ee578` was 1,006 passed,
14 skipped in 17.41 seconds.

## 3. Final Gated MLA and FFN correction

`add_ffn_after_final_global=True` is now the canonical default. The old
FFN-less final layer remains only as a standard-residual ablation; canonical
AttnRes configuration rejects it because every attention transformation must
be followed by a separately addressable FFN transformation.

## 4. Mathematical sources

The depth operator follows the equations in
[*Attention Residuals*](https://arxiv.org/abs/2603.15031):

```text
score_i = q^T RMSNorm_site(source_i)
weight  = softmax(score, over depth sources)
mixed   = sum_i weight_i * raw_source_i
```

There is no `sqrt(D)` factor, temperature, projection, head split, RoPE or
depth-position bias. The Kimi K3 phase specification determines the placement
before attention, before FFN and before the final RMSNorm. The
[official MoonshotAI reference](https://github.com/MoonshotAI/Attention-Residuals)
was used to check the separation between embedding, completed block sums and
the single current partial sum.

## 5. Transformer layers and depth sites

Each complete transformer layer contributes two independent sites:
`pre_attention` and `pre_ffn`. A separate `final_output` site is appended.
For `R` transformer layers:

```text
transformation sites = 2R
total sites          = 2R + 1
```

The mini one-group test profile has `R=5` and 11 sites. The representable K3
profile has `R=93`, 186 transformation sites and 187 total sites.

## 6. Full Attention Residuals

`mode="full"` retains the embedding at source index zero and then each raw
post-output-dropout attention and FFN output in execution order. Every site
mixes all sources currently registered. The final site mixes the embedding
and all 186 sublayer outputs in the 93-layer profile, then `final_norm` is
applied.

## 7. Block Attention Residuals

`mode="block"` retains the embedding separately, completed depth-block sums,
and at most one current partial sum. The partial starts from the first
sublayer output and never contains the embedding. Both block-size units are
explicit; 12 transformer layers correspond to 24 AttnRes sublayers.

For the K3 topology the state test observes:

```text
block sizes: 24, 24, 24, 24, 24, 24, 24, 18
final source count: 9 (embedding + 8 completed blocks)
```

Exact division creates no empty block, while partial final blocks are closed
exactly once.

## 8. State and final-output design

`FullAttentionResidualState` owns an ordered ephemeral list of sources.
`BlockAttentionResidualState` owns embedding, completed sums, one optional
partial, counters and temporary two-phase statistics. Both validate rank,
shape, dtype and device invariants and support non-aliasing clones.

The final output never selects the last sublayer output directly:

```text
ephemeral state -> independent final AttnRes site -> final RMSNorm
```

## 9. Parameters and initialization

Every site owns an independent zero-initialized pseudo-query `[D]` and affine
bias-free RMSNorm scale `[D]`. Thus each site adds `2D` parameters:

```text
mini:  2 * 8    * 11  = 176 expected and observed
K3:    2 * 7168 * 187 = 2,680,832 expected
```

Zero initialization gives an exact uniform distribution over available
sources and therefore an arithmetic mean, not a sum.

## 10. Dtype and numerical policy

FP64 and FP32 retain native precision. BF16/FP16 RMS statistics, optionally
the logits, and optionally the weighted sum accumulate in FP32; the mixed
state returns in the source dtype. The two-phase implementation preserves the
logit and weighted-sum policies independently and reproduces RMSNorm's
low-precision cast points.

Observed on CPU:

```text
FP64 manual oracle max abs error: 0.0
FP32 large/small stress:          finite outputs, weights and gradients
BF16 vs FP32 max abs error:       9.70e-3
BF16 vs FP32 mean abs error:      1.88e-3
```

CUDA BF16 tests are present and explicitly skipped when CUDA is unavailable.

## 11. Eager and two-phase backends

The eager backend stacks the currently available sources and is the reference
operator. The two-phase Block backend vectorizes the fixed inter-block scan
once for every depth block, then merges the optional current partial with a
stable online-softmax tuple `(max_logit, exp_sum, weighted_sum)`. Parameters
are shared because the backend changes only execution, not modules.

For the mini model with block size four, the two-phase diagnostic reports
three inter-block scans—one for each actual depth block—not one rescan per
site.

## 12. Integration and cache semantics

The AttnRes path replaces both standard residual additions:

```text
AttnRes(history) -> pre-norm -> Attention -> dropout -> append raw output
AttnRes(history) -> pre-norm -> FFN       -> dropout -> append raw output
```

The standard path is unchanged and owns no AttnRes parameters. KDA/MLA caches
have exactly one entry per attention layer in both paths. Depth state is
created afresh for each full, prefill or decode call, is absent from state
dicts and module attributes, and never enters `HybridBackboneCache`.

## 13. Typed traces and diagnostics

The optional trace distinguishes embedding, all pre-attention mixtures, raw
attention outputs, pre-FFN mixtures, raw FFN outputs and final mixture.
Per-site diagnostics include source count, entropy, normalized entropy,
embedding and most-recent weights, dominant source, retrieval distance,
completed-block mass and current-partial mass. A padded matrix plus validity
mask and source labels makes the complete depth-selection pattern inspectable.

Memory diagnostics distinguish ephemeral source payload from persistent
decode cache. For `B=2,T=6,D=8`:

```text
Full:    11 source tensors, 1,056 elements
Block-4: 4 source tensors,    384 elements, blocks [4, 4, 2]
```

## 14. Test coverage and observed equivalences

The 183 Attention Residual tests cover configuration and topology, primitive
manual math, initialization, dtype, Full/Block state invariants, deterministic
boundary matrices, online-softmax forward/gradcheck, integration, standard
ablation, full/prefill/decode and irregular streaming, every prefix,
right-padding, cache equality, all trainable gradients, serialization,
diagnostics, parameter counts, memory accounting, CPU BF16 and CUDA skips.

Fixed-seed FP64 observations:

```text
manual oracle vs site max abs:       0.0
Full vs Block-size-1 max abs:        0.0
Block eager vs two-phase max abs:    4.44e-16
full vs prefill max abs:             0.0
full vs token decode max abs:        4.44e-16
eager vs two-phase gradient max abs: 9.17e-14
changed-future prefix error:         0.0
```

Final CPU regression:

```text
1,186 passed, 17 skipped, 0 failed
1,203 collected
29.52 seconds
```

## 15. Benchmark

`benchmark_attention_residuals.py` covers 2, 4, 8 and 16 synthetic transformer
layers at sequence lengths 1, 64 and 256, plus integrated decode at contexts
64 and 256. It reports runtime, exact peak source payload, conceptual source
reads, two-phase inter-block scans, persistent cache elements/bytes and the
number of AttnRes cache entries.

A one-repeat CPU smoke run at `D=32`, 16 layers and `T=256` observed:

```text
standard:        0.97 ms,   65,536 source bytes
Full eager:     29.08 ms, 1,081,344 source bytes
Block eager:    10.44 ms,  294,912 source bytes
Block two-phase:10.79 ms,  294,912 source bytes, 8 inter-block scans
```

At a context of 256, every integrated mode had 9,184 persistent cache
elements (36,736 bytes) and zero AttnRes cache entries. Timings are
environment-dependent microbenchmarks and not claims about optimized kernels.

## 16. Red flags closed, limitations and next phase

The audit corrected the missing canonical final FFN default and its phase-6
parameter report. Tests explicitly prevent embedding leakage into partial
sums, hidden standard residual additions, shared queries/norms, normalized
values, hidden scaling, off-by-one blocks, double append, final-source loss,
two-phase rescans, cross-token mixing and persistent AttnRes cache growth. A
separate regression closes the independent two-phase BF16 dtype policies.
The full-suite run also exposed an inherited MLA portability defect: on this
PyTorch build, SDPA returns NaNs for fully masked rows and multiplication by a
zero mask cannot clear them. The MLA wrapper now uses `torch.where` to produce
exact zeros for invalid rows; its pre-existing all-masked and right-padding
tests pass.

The implementation is pure PyTorch and intentionally has no fused
CUDA/Triton kernel, distributed execution or later AttnRes variants. At the
end of phase 7, the next phase was:

```text
8_IMPLEMENTAR_STABLE_LATENT_MOE.md
```

Stable LatentMoE was subsequently implemented in phase 8; this statement is
retained only as the historical phase-7 boundary.
