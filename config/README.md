# Configuration profiles

The public training pipeline consumes three independent YAML files:

```text
config/data/*.yaml       -> build_dataloaders_from_yaml
config/kimi_k3/*.yaml    -> build_model_from_yaml
config/training/*.yaml   -> train_kimi_from_yaml
```

All parsers reject unknown keys. A typo is therefore an error, not an ignored
experiment setting.

## Recommended profile sets

| Target | Data | Model | Training |
|---|---|---|---|
| CPU syntax/smoke | `synthetic_cpu_smoke.yaml` | `cpu_tiny.yaml` | `cpu_yaml_smoke.yaml` |
| NVIDIA T4, ~15 GB usable | `tinystories.yaml` | `t4_15gb.yaml` | `t4_15gb.yaml` |
| 24 GB GPU | `tinystories_1024.yaml` | `gpu_24gb.yaml` | `gpu_24gb.yaml` |
| 48 GB GPU + PCC | `fineweb_edu_8192.yaml` | `gpu_48gb.yaml` | `gpu_48gb_pcc.yaml` |

The GPU profiles are conservative starting points, not universal memory
guarantees. Available memory also depends on PyTorch/CUDA versions, allocator
fragmentation, attention backend, dataset modality and diagnostics.

T4 intentionally uses FP16 because Turing GPUs do not provide the native BF16
path expected by the larger profiles.

## Responsibilities

- Data YAML owns corpus, tokenizer limits, block size, batch sizes, workers,
  shuffling and host-memory behavior.
- Model YAML owns every architecture dimension and capability.
- Training YAML owns precision, accumulation, MTP use, optimizer family,
  NTP/MTP loss settings, Muon/AdamW parameters, scheduler, diagnostics, PCC,
  EMA, checkpoints and previews.

`progressive_context_curriculum.yaml` remains a readable PCC reference
fragment. Standalone executable PCC configuration is
`training/gpu_48gb_pcc.yaml`.
