# Kimi-K3 Mini

A research-oriented, pure-PyTorch, reduced-scale implementation of the
principal architectural mechanisms introduced or combined in Kimi K3.

## Current status: Phase 0

The repository currently contains only a control model:

```text
token ids → embedding → causal MHA + RoPE → SwiGLU → RMSNorm → LM head
```

It is a baseline Transformer, **not** a paper-faithful Kimi-K3 implementation.
KDA, Gated MLA, Attention Residuals, Stable LatentMoE, MoonViT-V2, MTP, and RL
are intentionally absent until their corresponding phases.

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
