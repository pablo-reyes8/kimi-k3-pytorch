# Distributed inference and cache ownership

The public inference path remains `inference_autoregressive`; distributed
execution only changes parameter and cache ownership.

```bash
torchrun --standalone --nproc_per_node=2 \
  -m scripts.infer_kimi \
  --profile config/kimi_full_pipeline/distributed_tp_2x_24gb \
  --checkpoint checkpoints/consolidated_model.pt \
  --inference-config config/inference/distributed_greedy_tp2.yaml \
  --prompt "Once upon a time"
```

The baseline loads the same trusted consolidated checkpoint on every process
and then applies the TP transformation. It does not directly consume the
rank-sharded training checkpoint directory; TP/EP consolidation and elastic
resharding are intentionally not claimed.

## Cache layout

- Each TP rank owns complete local KDA recurrent heads and their three
  ShortConv histories.
- MLA Q/K/V heads are local, while the compressed latent cache, validity mask
  and sequence offsets are replicated.
- EP owns no persistent attention cache.
- DP ranks represent independent requests and never synchronize their caches.

Prefill and decode continue through `KimiK3` and
`HybridBackboneCache`. TP tests compare full forward, every prefix split,
token decode, right padding, clone/reorder behavior and local KDA cache
ownership against the unsharded model.

For readability at repository scale, the vocabulary head gathers logits on
TP ranks. This preserves greedy, temperature, top-k, top-p and repetition
penalty semantics but is a communication bottleneck, not a production serving
strategy.

