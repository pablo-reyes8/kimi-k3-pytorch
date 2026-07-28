# Dense FFN to Stable LatentMoE migration

Phase 8 preserves `DenseKimiFFN` as an explicit ablation while making
Stable LatentMoE the canonical channel mixer of `KimiBlock`.

The low-level `HybridBackboneConfig` remains backward compatible:

```python
channel_mixer_type="dense"
use_dense_ffn=True
```

The MoE profile is selected explicitly:

```python
channel_mixer_type="stable_latent_moe"
use_dense_ffn=False
stable_latent_moe_config=moe_config
```

Exactly one channel mixer is constructed per transformer layer. This includes
every KDA, periodic Gated MLA and the final global Gated MLA. Dense and MoE
mixers cannot be active in parallel.

Both mixers retain the tensor contract:

```text
[B,T,D] -> [B,T,D]
```

Under Attention Residuals, the complete MoE result
`shared_output + routed_output` is registered as one depth source. Individual
experts, latent activations and branch outputs are never registered as
separate depth sites. No standard residual is added around the MoE.

`KimiBlock` is the high-level phase-8 entrypoint. It makes the repeated
KDA/MLA pattern configurable while retaining the final global Gated MLA,
Stable LatentMoE after every attention layer, and Full/Block AttnRes. It
intentionally has no token embeddings, LM head, causal loss or generation API;
those belong to the final model orchestrator.
