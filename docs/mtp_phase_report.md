# Phase 9: single-layer MTP

This phase adds the auxiliary future-token head without introducing the final
Kimi K3 model orchestrator.

## Canonical alignment

For a sequence of length `T`, the only accepted training view is:

```text
source_hidden    = last_hidden_state[:, :-2]  # h[t]
future_input_ids = input_ids[:, 1:-1]         # x[t+1]
target_ids       = labels[:, 2:]               # x[t+2]
```

`build_mtp_training_view` centralizes this shift. A target is valid only when
the source, intermediate token and target are all unmasked, the target is not
`ignore_index`, and all three positions share a segment. The execution head
supports left and right padding by compacting valid structural triplets.
Packed multi-document execution is rejected because the current KDA/MLA cache
does not yet reset state per segment; silently leaking attention is forbidden.

## Architecture

`KimiMTPFusion` applies two independent RMSNorms, concatenates
`h[t]` with `embedding(x[t+1])`, and uses one bias-free `2D -> D` projection.
`KimiMTPBlock` then executes exactly one independent hybrid group:

```text
3 KDA + 1 causal Gated MLA
each followed by Stable LatentMoE
8 pre-sublayer Attention Residual sites
1 final local Attention Residual site
final RMSNorm
```

There is deliberately no additional global MLA. The input embedding and main
LM projection are module references supplied to `KimiMTPHead`; the LM weight
is therefore the real shared `Parameter`, not a copied tensor.

The MTP loss is the mean cross-entropy over its own valid targets. The combined
objective is `ntp_loss + loss_weight * mtp_loss`. `loss_weight=0.1` is a
project default and is not reported by the Kimi K3 paper.

## Checkpoint migration

MTP-disabled configurations instantiate neither fusion nor the auxiliary
hybrid group. When loading a pre-MTP checkpoint into a future orchestrated
model, use strict loading to detect missing MTP keys, or explicitly opt into
non-strict loading and initialize every reported MTP key. Do not suppress
missing keys silently. Reconstruct tied embedding/LM modules before loading so
their identity survives the round trip.

## Deferred scope

`draft_one_step` exposes an eval-only cache hook and `DraftFeatureProvider`
reserves a feature-source boundary. EAGLE-3 feature fusion, seven-step unroll,
acceptance-rate loss and speculative verification are intentionally deferred.
