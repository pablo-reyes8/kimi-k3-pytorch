# Phase 10 test report

Baseline before Phase 10:

```text
1350 collected
1329 passed
21 skipped
```

Architecture-only legacy cleanup removed 199 tests tied exclusively to the
unused baseline MHA, RoPE, dense SwiGLU, baseline Transformer stack,
BaselineCausalLM and legacy TokenEmbedding. No training, data, checkpoint,
optimizer, scheduler, seed, autocast or shared RMSNorm tests were removed.

Phase 10 adds 61 CPU tests covering:

```text
configuration and construction
text and inputs_embeds forwards
main/MTP parameter sharing
exact module topology
image, video and multiple-image composition
strict placeholder/count errors
full vs prefill/decode equivalence
serialization and deterministic outputs
BF16 CPU inference
typed outputs and absence of loss computation
```

Final regression:

```text
1212 collected
1197 passed
15 skipped
0 failed
```

The tiny real model has 50,068 unique parameters. Representative shapes are
text `[2,8]`, image `[2,3,16,16]`, logits `[2,8,128]` and MTP logits
`[2,6,128]`. Full versus cached comparisons use `atol=1e-6, rtol=1e-5`;
observed maximum discrepancies are below `2e-7`.
