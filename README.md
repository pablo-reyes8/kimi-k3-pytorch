# Kimi-K3 Mini

A research-oriented, pure-PyTorch, reduced-scale implementation of the
principal architectural mechanisms introduced or combined in Kimi K3.

## Current status: Phase 8

The repository now contains the research-scale text backbone through Attention
Residuals:

```text
3 KDA + 1 Gated MLA per hybrid group
→ additional final Gated MLA
→ Stable LatentMoE channel mixers
→ selectable standard, Full AttnRes, or Block AttnRes depth mixing
→ final RMSNorm
```

Both eager and exact two-phase Block AttnRes backends support full, prefill and
autoregressive decode with typed KDA/MLA caches. Stable LatentMoE provides
full-width shared experts, latent sparse routed experts, exact/histogram
Quantile Balancing and reference/vectorized dispatch. MTP, final multimodal
integration and the LM-level orchestrator remain future phases.

Generic Transformer, data, and training infrastructure was adapted from the
author's MIT-licensed DeepSeek-V4 Mini project. There are no runtime imports
from that repository.

## Batch contract

All datasets return `input_ids`, already-shifted next-token `labels`, and an
optional `attention_mask`. The model does not shift labels internally.

## Smoke test

```bash
pytest
python -m scripts.cpu_smoke_train
```

The synthetic retrieval data is self-contained and requires no downloads.

The full testing philosophy and invariant matrix are documented in
[`docs/testing.md`](docs/testing.md).
