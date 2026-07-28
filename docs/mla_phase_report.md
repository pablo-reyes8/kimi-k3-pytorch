# Kimi K3 Gated MLA phase report

## Mathematical source and scope

The implementation follows Kimi K3 Technical Report §2.1.2, equation 7. This
phase closes Gated MLA as an independent global-attention module. It does not
add the hybrid 3:1 backbone, Attention Residuals, MoE, or MTP.

No DeepSeek-specific cache or positional code was copied. The reusable
project-local `FullRankOutputGate` and head-layout helper are shared with KDA;
the MLA projections, attention cores, masks, cache, decoding, and diagnostics
are isolated under `src/mla`.

## Implemented equations

```text
q_t = W_q x_t
c_t = W_c x_t
k_t = W_k_up c_t
v_t = W_v_up c_t

P[t,s] = softmax(q_t k_s^T / sqrt(Q)), s <= t
o_t = sum_s P[t,s] v_s
y_t = W_o(sigmoid(W_g x_t) * combine_heads(o_t))
```

There is no RMSNorm or residual addition inside the module.

## Shapes

```text
hidden/final: [B,T,D]
q/k:          [B,T,H,Q]
v/raw:        [B,T,H,V]
latent cache: [B,T_cache,L]
D = H*V; Q and V may differ
```

## NoPE policy

MLA contains no rotary or absolute embedding, relative-position bias, or
position-dependent query/key transform. `position_ids` is deliberately rejected
instead of being silently ignored. KDA will provide position-sensitive and
recency-aware mixing once both operators are composed.

## Precision policy

When `keep_attention_output_fp32=True`, FP16/BF16 q/k/v are explicitly cast for
FP32 score, softmax, and value accumulation. The ungated attention output stays
FP32 through channel gating and the output linear operation; only the public
final activation is cast back to model dtype. FP32 and FP64 keep their native
precision. Both manual and PyTorch SDPA backends implement the same policy.

## Latent cache

`MLACache` stores only compressed `W_c x`, a boolean validity mask, and valid
lengths. K/V are reconstructed from the complete latent sequence when needed.
The cache is functional (not mutated), cloneable, reorderable, serializable,
and compacts right-padded batches before appending new chunks.

Per-token storage is:

```text
MLA: L
full multi-head KV: H*(Q+V)
compression ratio: H*(Q+V)/L
```

For the test configuration `(H,Q,V,L)=(3,2,4,5)`, the full-KV/latent ratio is
`18/5 = 3.6x`.

## Verification

The MLA suite covers configuration failures, exact projection equations,
manual scalar attention, manual-vs-SDPA forward and gradients, causal diagonal
and prefix invariants, NoPE permutation equivariance, uniform and near-one-hot
attention, full-rank gating, every cache split point, token decode, irregular
streaming, right-padding with unequal batch lengths, cache reorder/reset,
gradients for every architectural parameter, four gradchecks, FP32/BF16,
dropout, diagnostics, and state/cache serialization.

CPU result:

```text
tests/mla: 107 passed, 2 skipped (CUDA-only)
full repository: 910 passed, 13 skipped
```

At float64 test dimensions, the measured manual-vs-SDPA maximum output error
was `0.0`; maximum input-gradient and parameter-gradient errors were
`1.62e-27` and `1.65e-24`. Full-vs-token-decode maximum error was `1.08e-19`.
On CPU BF16, against the same FP32 module/input, maximum absolute and mean
absolute errors were `1.03e-6` and `1.29e-7` (maximum relative error `5.46e-2`,
dominated by near-zero outputs). CUDA BF16 tests are present and skip cleanly
when CUDA is unavailable.

## Known limitations

- The reference implementation reconstructs full K/V from latent cache for each
  decode call; this is faithful but not a fused production kernel.
- Only monotonic right-padding is a public module contract.
- Attention is quadratic and intended as the periodic global component.
- CUDA-specific rounding metrics were not executed on the CPU-only test host.

The next phase is `6_CONSTRUIR_HYBRID_ATTENTION_BACKBONE.md`; it should combine
three KDA layers with one Gated MLA layer without changing either closed
operator.
