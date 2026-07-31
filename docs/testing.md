# Testing standard

Tests are part of the architectural specification of Kimi-K3 Mini. A module is
not considered correct merely because it returns finite values or the expected
shape.

## Required test dimensions

Every mathematical module should cover, where applicable:

1. configuration validation and precise failure modes;
2. parameter shapes, initialization, tying, and bias policy;
3. a direct comparison with the reference equation;
4. tensor shape, dtype, device, and empty-dimension behavior;
5. invariants such as causality, normalization, norm preservation, or masking;
6. deterministic versus stochastic train/eval behavior;
7. gradients for inputs and every trainable parameter;
8. BF16 behavior and optional CUDA/FP16 behavior;
9. state-dictionary or checkpoint roundtrips;
10. composition with adjacent modules;
11. a small download-free synthetic-data path;
12. regression tests for every discovered bug.

Finite-value assertions remain useful, but only as one check within a stronger
mathematical or behavioral test.

## Current coverage matrix

| Area | Principal guarantees |
|---|---|
| Token embedding | validation, initialization statistics, padding, scaling, dropout, gradients, tying surface, BF16, synthetic integration |
| RMSNorm | exact FP32 reference equation, last-axis behavior, extremes, scale invariance, gradients, BF16 |
| RoPE | exact rotation equation, norm and dot-product invariants, partial rotation, position forms, gradients, BF16 |
| MHA baseline | manual attention equation, exact causal zeros, padding masks, fully masked rows, RoPE/NoPE behavior, cached decoding equivalence |
| SwiGLU | exact activation equation, dimension resolution, initialization, dropout, gradients, BF16 |
| Block and stack | pre-norm residual equation, identity under zero sublayers, causal composition, mask/position propagation, layer independence |
| Causal LM | strict output contract, shifted-label loss equation, auto masks, embedding entrypoint, tying, causality, complete gradients |
| Synthetic retrieval | tokenizer, metadata consistency, exact label offsets, answer retention, deterministic indexing, split isolation, MTP offsets |
| Text data | local tokenizer training/loading, exact causal blocks, preset/config forwarding; no network calls |
| Training | AdamW/Muon grouping, Newton–Schulz behavior, scheduler equations, AMP, EMA, checkpoints/RNG, accumulation, eval, overfit |
| Distributed | strict topology/env parsing, rank mapping, serializable samplers, token-exact DDP update parity, complete-head TP forward/backward/cache parity, no-drop EP output/gradient/global-QB parity, TP×EP smoke and atomic same-topology resume |
| MoonViT visual | exact patch/attention/MLP equations, positional interpolation, masks, bias policy, RMSNorm, dynamic grids, pixel packing, projector, gradients, BF16, roundtrips, synthetic spatial overfit |
| Hierarchical visual ablation | pooling equation/masks, stage grids, global-attention diagnostics, dynamic resolution, gradients, BF16 |
| Swin visual ablation | exact window roundtrips, shifted-window masks, relative positions, patch-merging order, odd grids, gradients, BF16 |
| Kimi primitives | SiTU bounds/equation/gradcheck, causal ShortConv full-cache equivalence, head-wise RMS independence, full-rank gate source/saturation, typed states, exact head reshapes, BF16 |
| Kimi Delta Attention | K3 projections/decay, recurrent equations/invariants, UT solve and WY auxiliaries, chunk-size/route/gradient equivalence, prefix causality, masks, constant-size decode cache, BF16 and serialization |
| Kimi Gated MLA | NoPE inspection/equivariance, shared latent KV equations, manual scalar/SDPA parity, causal global attention, full-rank gate, compressed-cache prefill/decode, unequal right-padding, complete gradients/gradchecks, BF16 and serialization |
| Hybrid attention backbone | explicit repeated 3:1 KDA/MLA pattern plus final MLA, two pre-norm residual branches, replaceable dense SiTU-GLU FFN, typed synchronized caches, full/prefill/decode and unequal-padding equivalence, end-to-end causality, parameter counts, gradients/gradcheck and serialization |
| Attention Residuals | exact depth-softmax oracle, Full/Block state machines, block-size-1 parity, eager/two-phase online-softmax parity, independent site parameters, final mixer, no residual leakage, full/prefill/decode and cache equivalence, causal/padding invariants, gradients/gradcheck, BF16, diagnostics, memory accounting and serialization |
| Stable LatentMoE | full-width shared and latent routed expert equations, sigmoid biased Top-k with unbiased weights, reference/vectorized sparse dispatch and gradient parity, exact/histogram Quantile Balancing, causal bias commits, zero-load experts, BF16, serialization, routing diagnostics, KDA/MLA/AttnRes integration and KimiBlock full/prefill/decode |

## CPU command

```bash
python3 -m pytest
```

CUDA tests must use `skipif` and must never make the CPU suite fail merely
because a GPU is unavailable. Network downloads are forbidden in unit tests.

## Regression policy

When a test exposes a real defect:

1. keep the failing test;
2. fix the smallest relevant production surface;
3. rerun the complete suite;
4. record any altered contract;
5. do not weaken numerical tolerances merely to obtain a pass.
