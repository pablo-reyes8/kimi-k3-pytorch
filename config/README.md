# Configuration profiles

The public training pipeline consumes one self-contained profile directory:

```text
config/kimi_full_pipeline/<profile>/
├── data.yaml       -> build_dataloaders_from_yaml
├── model.yaml      -> build_model_from_yaml
└── training.yaml   -> train_kimi_from_yaml
```

All parsers reject unknown keys. A typo is therefore an error, not an ignored
experiment setting.

## Recommended profile sets

| Directory | Target | Default data |
|---|---|---|
| `cpu_smoke/` | CPU syntax/smoke | Synthetic retrieval |
| `low_gpu/` | NVIDIA T4, about 15 GB usable | WikiText-2 |
| `gpu_24gb/` | 24 GB research GPU | FineWeb 10BT streaming |
| `gpu_48gb/` | 48 GB GPU + PCC | FineWeb 100BT streaming |
| `gpu_80gb/` | 80 GB GPU + PCC | FineWeb 350BT streaming |
| `canonical/` | Full architecture/training metadata | FineWeb 350BT streaming |
| `distributed_ddp_2x_t4/` | 2-rank DDP | Synthetic retrieval |
| `distributed_tp_2x_24gb/` | 2-rank Kimi TP | FineWeb 10BT streaming |
| `distributed_tp_ep_4x_24gb/` | 4-rank TP × EP | FineWeb 10BT streaming |

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
  EMA, checkpoints, previews and the strict DP×TP×EP topology.

Legacy component and reference fragments remain in their specialized config
directories. They are not complete pipeline profiles. See
`kimi_full_pipeline/README.md` for the new standard.

Generation-time greedy and sampling profiles live under `config/inference/`.
Standalone WikiText-2 and progressively scaled FineWeb data YAMLs live under
`config/data/`; the FineWeb variants always default to streaming plus explicit
document caps.
