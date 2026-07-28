# Phase 0 migration audit

The source repository was inspected only in `data/`, `src/`, and `training/`,
apart from its MIT license. Kimi-K3 Mini owns all migrated code and has no
runtime dependency on the DeepSeek-V4 Mini checkout.

| Original file | New file | Action | Coupling removed | Tests |
|---|---|---|---|---|
| `src/transformer_modules/embedding_module.py` | `src/transformer_modules/embedding.py` | Adapted | Project naming; generic API retained | Passed |
| `src/transformer_modules/RMSNorm.py` | `src/transformer_modules/rms_norm.py` | Copied/renamed | None required | Passed |
| `src/transformer_modules/mha_baseline.py` | `src/transformer_modules/mha.py` | Adapted | Baseline explicitly separated from Gated MLA | Passed |
| `src/transformer_modules/rope*.py` | same generic names | Adapted | RoPE remains optional and baseline-only | Passed |
| `src/transformer_modules/SwiGLU.py` | `src/transformer_modules/swiglu.py` | Adapted | Renamed generic feed-forward API | Passed |
| `src/transformer_modules/transformer_block.py` | `src/transformer_modules/baseline_block.py` | Adapted | Explicit control-model identity | Passed |
| `src/transformer_modules/transformer.py` | `src/causal_lm.py` | Rebuilt | Large output dict and shifted-label ambiguity | Passed |
| `data/data_utils.py` | `data/batch.py` | Adapted | English contract; shifted labels documented | Passed |
| `data/syntethic_long_context_retrieval.py` | `data/synthetic_long_context_retrieval.py` | Adapted/renamed | Architecture names; ensured query survives tiny contexts | Passed |
| `data/text_datasets.py` | `data/text_datasets.py` | Copied | Already generic | Import passed |
| `training/autocast.py` | `training/autocast.py` | Adapted | Model-specific examples removed | Passed |
| `training/adam_optmizer.py` | `training/adam_optimizer.py` | Rebuilt/renamed | mHC/MTP name-based grouping removed | Passed |
| `training/chekpoints.py` | `training/checkpoints.py` | Rebuilt/renamed | Generic payload and naming | Passed |
| `training/muon_optimizer.py` | `training/muon_optimizer.py` | Adapted | DeepSeek-specific exclusions removed | Import passed |
| `training/scheduler.py` | `training/scheduler.py` | Adapted | Project naming only | Import passed |
| `training/seed.py` | `training/seed.py` | Copied | None required | Import passed |
| `training/ema.py` | `training/ema.py` | Adapted | Project naming only | Import passed |
| `training/train_one_epoch.py` | `training/train_one_epoch.py` | Rebuilt | MTP/MoE/mHC diagnostics replaced by small output contract | Passed |
| `training/eval_one_epoch.py` | `training/eval_one_epoch.py` | Rebuilt | DeepSeek diagnostics and preview coupling removed | Import passed |
| `training/training_metrics.py` | `training/training_metrics.py` | Rebuilt | Model-specific output probing removed | Import passed |

## Counts

- Copied unchanged except conventional placement/filename: 3
- Adapted or rebuilt: 18
- Intentionally excluded from the active production path: 20
- Unit tests: 351 passing, 7 hardware-gated CUDA skips
- CPU smoke training: 8 batches and 8 optimizer steps, finite mean loss

## Intentionally excluded

The following DeepSeek-specific source modules were not migrated: CSA and its
indexer/components, HCA and its components, mHC residuals, DeepSeek block/model
assembly, DeepSeek MoE, and MTP assembly/components. Model-specific mHC, MoE,
MTP, and full-DeepSeek metrics were also excluded. `tinystories_data.py` was
not copied because the generic `text_datasets.py` already exposes TinyStories
without retaining the separate MTP path.

## Known TODOs

- Per-Head Muon is reserved for Phase 6.
- Kimi-specific output diagnostics will be added with their actual modules.
- Hugging Face datasets/tokenizers remain optional dependencies and were not
  downloaded during Phase 0 verification.
- No generation cache exists in the Phase 0 baseline.

## Test-hardening follow-up

The initial smoke suite was replaced with equation-level and behavioral tests
for every active source, data, and training component. This work found and
fixed two real Phase 0 defects:

- synthetic map-style samples were stateful and ignored their dataset index;
- gradient accumulation dropped a final partial accumulation window.

See `docs/testing.md` for the required standard for future Kimi modules.

## Fidelity boundary

The active model is a standard Transformer control. KDA, Gated MLA, Attention
Residuals, Stable LatentMoE, SiTU-GLU, Quantile Balancing, MoonViT-V2, MTP,
and RL are not implemented.
