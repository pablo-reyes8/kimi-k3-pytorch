# Kimi K3 component map

This directory is a documentation index. Implementations live in focused
packages under `src/`; `src/kimi_k3_mini.py` is the single public model class
that composes them.

## Execution path

```text
token IDs ──> embedding ──────────────┐
                                      ├─> shared Kimi backbone ─> LM head
images/videos -> MoonViT -> projector ┘              │
                                                     └─> auxiliary MTP head
```

The shared sequence contains visual embeddings in reserved placeholder
positions. Text and vision then use the same KDA/MLA, MoE and LM-head path.

## Components

### Kimi Delta Attention

Location: `src/kda/`

- recurrent, chunkwise and decode execution;
- short-convolution state;
- data-dependent alpha/beta and decay state;
- NoPE operation;
- FP32 state accumulation and diagnostics.

### Gated Multi-head Latent Attention

Location: `src/mla/`

- compressed KV latent representation;
- global causal attention;
- output gate;
- NoPE operation;
- manual/SDPA backends and cache support.

### Hybrid backbone

Locations: `src/hybrid_backbone/`, `src/kimi_block.py`

- configurable repeated attention pattern;
- canonical `3 KDA + 1 Gated MLA`;
- additional final Gated MLA;
- one Stable LatentMoE channel mixer after every attention layer;
- typed prefill/decode caches.

### Attention Residuals

Location: `src/attention_residuals/`

- Full and Block AttnRes;
- eager and exact two-phase block execution;
- depth-source mixing and numerical diagnostics;
- embedding source plus final output mixer.

### Stable LatentMoE

Location: `src/stable_latent_moe/`

- full-width shared experts;
- latent routed experts with SiTU-GLU;
- reference/vectorized dispatch;
- exact/histogram Quantile Balancing;
- routing bias stored as state, not a trainable parameter;
- router/load/branch diagnostics.

### Vision path

Location: `src/vision/`, integration in `src/kimi_k3/`

- MoonViT, hierarchical and Swin variants;
- optional spatial pixel shuffle;
- projector into `d_model`;
- image/video placeholder replacement;
- text-only execution without hidden visual modules.

### Multi-Token Prediction

Location: `src/mtp/`

- one auxiliary group predicting `x[t+2]`;
- normalized hidden/future-embedding fusion;
- shared main LM-head parameter;
- no effect on main logits or cached decode.

### Objectives

Location: `src/loss/`

- next-token pretraining CE;
- MTP auxiliary objective;
- SFT, policy optimization and distillation objectives kept explicit for
  future post-training.

### Master model and configuration

Locations: `src/kimi_k3_mini.py`, `src/kimi_k3/`

- `KimiK3Config` validates the fully composed architecture;
- `KimiK3` owns the public forward/prefill/decode contract;
- `build_model_from_yaml` builds the same typed config from YAML;
- no component mathematics is duplicated in the master class.

## Training ownership

Training-only behavior does not live in `src/`:

- optimizers, Muon and QK-Clip: `training/optimizer/`;
- Quantile Balancing commit timing: `training/moe_control.py`;
- diagnostics and structured printing: `training/diagnostics/`;
- Progressive Context Curriculum: `training/context_curriculum/`;
- master training entrypoint: `training/train_kimi_k3.py`;
- YAML training adapter: `training/yaml_config.py`.

For CLI/Jupyter, use the three-YAML pipeline documented in
`docs/yaml_training_pipeline.md`. Lower-level components remain public for
tests and research, but user-facing training should enter through
`train_kimiK3` or `train_kimi_from_yaml`.
