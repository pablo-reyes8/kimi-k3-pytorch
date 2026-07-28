# Kimi K3 hybrid attention backbone phase report

## Architecture and scope

The backbone composes the already-closed KDA and Gated MLA operators as:

```text
(KDA -> dense FFN) x3
-> (Gated MLA -> dense FFN)
-> repeat by group
-> final Gated MLA
-> final RMSNorm
```

The canonical 3:1 pattern and final global layer are explicit and inspectable
through `model.attention_types`. This phase intentionally contains neither
Attention Residuals, Stable LatentMoE, Quantile Balancing, MTP, nor visual
integration.

## Depth and channel interfaces

Every regular layer owns two independent pre-norm residual branches:

```text
u = x + dropout(attention(RMSNorm(x)))
y = u + dropout(ffn(RMSNorm(u)))
```

The final global MLA is also paired with an FFN in the corrected canonical
profile; omitting it remains available only as a standard-residual ablation.
`ffn` is treated only as `nn.Module: [B,T,D] -> [B,T,D]`; the current `DenseKimiFFN` wraps SiTU-GLU
and can later be replaced by Stable LatentMoE without editing KDA, MLA, cache
or attention routing.

Hidden-state capture returns the input, every post-layer residual state before
the final norm, and the final normalized output. These named depth boundaries
are the future attachment surface for Attention Residuals and MTP.

## Hybrid cache

There is exactly one typed cache per attention layer, in model order:

- KDA layers store recurrent state, three ShortConv histories and offsets.
- MLA layers store compressed latent KV, validity mask and offsets.
- The additional final MLA has its own independent latent cache.

`HybridBackboneCache` validates synchronized per-sample offsets while retaining
a scalar maximum sequence length for accounting. It supports clone, batch
reorder, serialization, unequal right-padding and functional updates without
mutating the supplied prefix.

## Precision correction

The baseline RMSNorm policy was made consistent with KDA/MLA primitives:
FP16/BF16 accumulate in FP32, while FP32 and FP64 retain native precision.
Previously, forcing FP64 through FP32 prevented numerical gradcheck of the
composed backbone.

## Verification

Tests cover the exact repeated pattern, final MLA, metadata, independent
parameter storage, analytical parameter counts, explicit residual equations,
no double residual, generic FFN replacement, manual layer iteration, all
prefill split points, token decode, irregular chunk streaming, equality of
every KDA/MLA cache, unequal padded prefill followed by decode, end-to-end
causality for every prefix, complete gradients, full gradcheck, diagnostics,
dropout, state/cache serialization, stress inputs and BF16.

CPU verification:

```text
tests/hybrid_backbone: 96 passed, 1 skipped (CUDA-only)
full repository:       1006 passed, 14 skipped
```

For the deterministic float64 validation sequence:

```text
full vs prefill max abs:       0.0
full vs token decode max abs:  5.55e-17
final cache max abs:           2.78e-17
changed-future prefix error:   0.0
```

CPU BF16 versus FP32 produced maximum/mean absolute errors of `9.45e-3` and
`1.86e-3`, with maximum relative error `9.01e-3`. CUDA BF16 coverage is
present but skipped on the CPU-only host.

The corrected one-group canonical profile contains 3,030 parameters:

```text
KDA attention:  1,062
Gated MLA:        440
dense FFN:      1,440
RMSNorms:          88
```

Basic CPU benchmark at `D=32`, `T=64`:

```text
groups  full ms  prefill ms  decode ms/token  cache elements
1        16.05      16.66          5.36          3,008
2        40.31      42.35         12.61          4,992
4        83.41      84.82         24.54          8,960
```

These are correctness-oriented pure-PyTorch numbers, not optimized kernel
targets. The benchmark also records exact cache bytes and process peak RSS; the
one-group run reported 12,032 cache bytes and approximately 201 MB total
process peak RSS (including Python and PyTorch runtime).

The next phase is `7_IMPLEMENTAR_ATTENTION_RESIDUALS.md`. AttnRes should consume
the recorded post-layer depth states without changing the sequence-mixing
operators or their typed caches.
