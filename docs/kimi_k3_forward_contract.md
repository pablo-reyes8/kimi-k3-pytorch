# KimiK3 forward contract

The public master class is `src.kimi_k3_mini.KimiK3`. That file defines only
the orchestrator. Immutable configuration, typed outputs, visual placeholder
composition, validation and visual adaptation live under `src/kimi_k3/`.

## Forward order

```text
input_ids or inputs_embeds
→ token embeddings
→ optional MoonViT / hierarchical / Swin encoder
→ optional 2×2 spatial token packing
→ visual projector
→ exact placeholder replacement
→ KimiBlock
   → repeated 3 KDA + 1 Gated MLA groups
   → Stable LatentMoE after every attention
   → Attention Residuals across depth
   → final global Gated MLA + Stable LatentMoE
   → final AttnRes + RMSNorm
→ shared or untied LM head
→ optional independent MTP logits
```

The main output shape is `[B, T, vocab_size]`. MTP, when explicitly requested
with `use_mtp=True`, returns `[B, max(T-2, 0), vocab_size]`. It is forbidden in
prefill/decode and cannot feed back into the main logits.

Phase 10 computed no NTP, MTP or total loss. Since phase 11, calls without
labels preserve that inference contract, while labeled pretraining calls
delegate to `src.loss.KimiPretrainingLoss`. SFT, RL, and MOPD remain explicit
trainer-side calls through `KimiTrainingObjective`.

## Multimodal engineering decisions

The report does not prescribe a public placeholder API. This implementation
uses exact replacement: every valid projected visual token requires one
reserved placeholder. `image_counts` and `video_counts` assign flattened
visual items to text samples. Counts and placeholder totals are validated per
sample; tokens are never truncated, repeated or borrowed across examples.

Images use `[M,C,H,W]`. Videos use `[M,F,C,H,W]`, encode frames through the
same vision encoder and concatenate their projected tokens per video.

## Cache contract

`prefill` and `decode_step` call the same master forward. Cached decode accepts
exactly one current token and a current-token boolean mask. Vision is consumed
only during prefill. MTP is full-sequence-only.

`position_ids` are rejected because the implemented KDA and Gated MLA path is
NoPE. Left padding is also rejected explicitly by the current hybrid cache;
right padding is supported.
