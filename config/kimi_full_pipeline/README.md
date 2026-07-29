# Full Kimi pipeline profiles

Each directory is one complete experiment control plane:

```text
profile/
├── data.yaml
├── model.yaml
└── training.yaml
```

Change only `PROFILE_DIR` in Jupyter, or pass the directory to the CLI. The
three existing public builders still consume the individual paths.

| Profile | Intended scope | Context |
|---|---|---:|
| `cpu_smoke` | Syntax and one-step CPU smoke | 32 |
| `low_gpu` | Conservative NVIDIA T4 / about 15 GB usable | 512 |
| `gpu_24gb` | 24 GB research GPU | 1,024 |
| `gpu_48gb` | 48 GB GPU with PCC | 8,192 |
| `gpu_80gb` | Large 80 GB research GPU with PCC | 8,192 |
| `canonical` | Full reported topology and training metadata | 8,192 |

GPU memory depends on backend, PyTorch/CUDA versions, allocator state,
diagnostic cadence and modality. `canonical` is metadata for validation and
future distributed orchestration; the current trainer is single-process and
must not instantiate it casually.

Inference sampling profiles remain in `config/inference/` because they are
orthogonal to data/model/training experiment selection.
