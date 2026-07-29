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

| Profile | Intended scope | Data | Context |
|---|---|---|---:|
| `cpu_smoke` | Syntax and one-step CPU smoke | Synthetic retrieval | 32 |
| `low_gpu` | Conservative T4 / about 15 GB usable | WikiText-2 | 512 |
| `t4_wikitext` | 212.9M T4 target with a language corpus | WikiText-2 | 1,024 |
| `t4_retrieval` | 246.6M T4 target with progressive context | Synthetic retrieval | 2,048 |
| `gpu_24gb` | 24 GB research GPU | FineWeb 10BT | 1,024 |
| `gpu_48gb` | 48 GB GPU with PCC | FineWeb 100BT | 8,192 |
| `gpu_80gb` | Large 80 GB research GPU with PCC | FineWeb 350BT | 8,192 |
| `canonical` | Full topology/training metadata | FineWeb 350BT | 8,192 |

GPU memory depends on backend, PyTorch/CUDA versions, allocator state,
diagnostic cadence and modality. `canonical` is metadata for validation and
future distributed orchestration; the current trainer is single-process and
must not instantiate it casually.

The two larger T4 profiles are allocation targets, not measured VRAM claims.
They use FP16, microbatch 1, gradient accumulation, no EMA and the hybrid
Muon/AdamW optimizer to preserve memory headroom. Run `--validate-only` first
and measure the initial allocation before committing to a long job.

FineWeb profiles use Hugging Face streaming and explicit document caps. Their
source-scale names describe the remote corpus, not an automatic promise to
materialize the full 10B/100B/350B-token sample.

Inference sampling profiles remain in `config/inference/` because they are
orthogonal to data/model/training experiment selection.
