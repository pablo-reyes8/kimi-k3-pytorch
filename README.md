
<p align="center">
  <img src="assets\header_image.png" width="1000"/>
</p>


## A from-scratch, research-scale PyTorch implementation of Kimi K3

[![CI](https://github.com/pablo-reyes8/kimi-k3/actions/workflows/ci.yml/badge.svg)](https://github.com/pablo-reyes8/kimi-k3/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active_research-orange.svg)](#project-status)

Kimi-K3 Mini is a pure-PyTorch implementation of the principal architectural,
training and inference mechanisms described or combined in Kimi K3. It scales
the system down into readable modules that can be validated on CPU, composed
from strict YAML profiles and progressively enlarged for real experiments.

This is not a generic Transformer renamed after Kimi. The repository includes
Kimi Delta Attention, Gated Multi-head Latent Attention, the hybrid 3:1
backbone, Stable LatentMoE, Attention Residuals, MoonViT integration, MTP,
Kimi-aware optimization, progressive context training and native cached
autoregressive inference.

> [!IMPORTANT]
> This is an independent research implementation. It is not affiliated with
> Moonshot AI, does not ship official or trained weights, and does not reproduce
> the original distributed training system, data mixture or production kernels.

## Contents

- [Why this repository exists](#why-this-repository-exists)
- [Implementation status](#implementation-status)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Complete YAML profiles](#complete-yaml-profiles)
- [Supported data](#supported-data)
- [Training](#training)
- [Inference](#inference)
- [Testing and CI](#testing-and-ci)
- [Docker](#docker)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Project status](#project-status)
- [Citation and license](#citation-and-license)

## Why this repository exists

Large-model reports are easiest to understand when their mechanisms can be
isolated, inspected and tested. This project provides:

- paper-oriented modules instead of one opaque model file;
- exact behavioral tests for equations, causality, masks, caches and gradients;
- tiny CPU configurations for development and larger research profiles;
- one public training orchestration path;
- one public autoregressive inference path;
- explicit boundaries between implemented research code and future systems
  work.

The canonical topology is represented as validated metadata, while smaller
profiles make the same composition practical for local experimentation.
The architectural reference bundled with the repository is
[*Kimi K3: Open Frontier Intelligence*](paper/k3_tech_report.pdf).

## Implementation status

| Area | Included |
|---|---|
| Kimi Delta Attention | Recurrent, chunkwise, prefill and decode paths; short-convolution state; data-dependent decay; FP32 accumulation |
| Gated MLA | NoPE global attention, compressed latent KV, full-rank output gate, manual/SDPA backends and cache |
| Hybrid backbone | Repeated `3 KDA + 1 Gated MLA`, final global MLA and synchronized heterogeneous cache |
| Stable LatentMoE | Shared experts, latent routed experts, sparse top-k dispatch, exact/histogram Quantile Balancing |
| Attention Residuals | Full and Block AttnRes with eager and exact two-phase execution |
| Vision | MoonViT, hierarchical and Swin variants, pixel shuffle, projection and image/video token composition |
| MTP | One auxiliary `x[t+2]` prediction group with normalized fusion and shared LM head |
| Objectives | NTP, MTP, trajectory SFT, policy optimization and multi-teacher on-policy distillation |
| Training | Train/eval epochs, AMP, accumulation, EMA, checkpoints, scheduler, previews and structured diagnostics |
| Kimi optimizers | AdamW, Muon/AdamW hybrid, per-head QKV handling and QK-Clip |
| Long-context curriculum | Optional Progressive Context Curriculum with resumable stage state |
| Inference | Checkpoint restoration, greedy/sampling generation and native KDA/MLA cached decode |
| Configuration | Strict data/model/training YAMLs grouped into complete experiment profiles |

The full component map is available in
[`src/kimi_components/README.md`](src/kimi_components/README.md).

## Architecture

```text
text token IDs ──> token embeddings ──────────────┐
                                                  │
images/videos ──> MoonViT ─> projector/composer ──┤
                                                  ▼

                                    repeated hybrid groups
                                  [3 × KDA + 1 × Gated MLA]
                                            │
                                Stable LatentMoE after attention
                                            │
                                Full or Block Attention Residuals
                                            │
                                  final global Gated MLA + MoE
                                            │
                                    RMSNorm + tied LM head
                                      ┌─────┴─────┐
                                  NTP logits   MTP x[t+2]
```

The canonical metadata profile describes:

- `d_model = 7168`;
- 56 attention heads;
- 23 hybrid groups plus the final Gated MLA, for 93 attention layers;
- 896 routed experts with 16 selected per token;
- a 160,000-token model vocabulary;
- MoonViT visual encoding and one MTP group.

Loading the canonical YAML validates this topology without allocating its
weights. Do not instantiate it casually.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/pablo-reyes8/kimi-k3.git
cd kimi-k3
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,data]"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,data]"
```

The minimal architecture/inference dependencies are installed with
`pip install -e .`. The `data` extra adds Hugging Face datasets and
tokenizers; `dev` adds pytest.

## Quick start

Validate a complete low-GPU experiment without downloading data, allocating a
model or training:

```bash
python -m scripts.train_kimi \
  --profile config/kimi_full_pipeline/low_gpu \
  --validate-only
```

The same operation through the Makefile:

```bash
make validate PROFILE=config/kimi_full_pipeline/low_gpu
```

Interactive examples:

- [`notebooks/train_kimi_k3_from_yaml.ipynb`](notebooks/train_kimi_k3_from_yaml.ipynb)
- [`notebooks/inference_kimi_k3_from_checkpoint.ipynb`](notebooks/inference_kimi_k3_from_checkpoint.ipynb)

## Complete YAML profiles

The public configuration standard is one self-contained directory:

```text
config/kimi_full_pipeline/<profile>/
├── data.yaml
├── model.yaml
└── training.yaml
```

| Profile | Intended target | Data source | Maximum context |
|---|---:|---|---:|
| `cpu_smoke` | CPU | Synthetic retrieval | 32 |
| `low_gpu` | T4 / about 15 GB | WikiText-2 | 512 |
| `gpu_24gb` | 24 GB GPU | FineWeb 10BT sample | 1,024 training / 2,048 data |
| `gpu_48gb` | 48 GB GPU | FineWeb 100BT sample | 8,192 |
| `gpu_80gb` | 80 GB GPU | FineWeb 350BT sample | 8,192 |
| `canonical` | Distributed metadata | FineWeb 350BT sample | 8,192 |

GPU memory depends on backend, PyTorch/CUDA versions, modality, allocator
state and diagnostics. These are starting points, not universal guarantees.

Use one resolver to obtain all three paths:

```python
from configuration import resolve_kimi_pipeline_profile

profile = resolve_kimi_pipeline_profile(
    "config/kimi_full_pipeline/low_gpu"
)
print(profile.data, profile.model, profile.training)
```

Unknown YAML fields fail loudly so configuration typos cannot silently change
an experiment.

## Supported data

The data orchestrator owns tokenization, causal blocks, loaders and the
context-aware loader factory used by PCC.

| Family | Presets |
|---|---|
| Local synthetic | Deterministic long-context key/value retrieval |
| Compact Hugging Face text | WikiText-2, TinyStories, AG News, IMDB, MiniPile |
| Educational web | FineWeb-Edu 10BT-mincols |
| Progressive LLM scale | FineWeb `sample-10BT`, `sample-100BT`, `sample-350BT` |

Hugging Face profiles can cap tokenizer, train and validation documents and
cache a byte-level BPE tokenizer. The three FineWeb profiles enable streaming
and document caps by default so selecting one does not eagerly download the
27.6 GB, 277.4 GB or roughly 388 GB remote source. Unit tests never download
datasets.

Standalone YAMLs for these sources live under [`config/data/`](config/data/).
Removing the document caps is an explicit large-scale operation and should
only be done with a deliberate storage and preprocessing plan.

Build a configured data bundle:

```python
from data import build_dataloaders_from_yaml

data = build_dataloaders_from_yaml(profile.data)
train_loader = data.train_loader
val_loader = data.val_loader
```

For inference, `load_tokenizer_from_data_yaml` reconstructs the deterministic
synthetic tokenizer or loads the cached tokenizer without rebuilding loaders.

## Training

The notebook and CLI use three public calls in order:

```python
from data import build_dataloaders_from_yaml
from src import build_model_from_yaml
from training import train_kimi_from_yaml

data = build_dataloaders_from_yaml(profile.data)
model = build_model_from_yaml(profile.model, data_bundle=data)
history = train_kimi_from_yaml(
    profile.training,
    model=model,
    data=data,
)
```

`train_kimi_from_yaml` validates cross-file invariants and delegates to the
single master orchestrator, `train_kimiK3`. The training YAML controls:

- precision, epochs, accumulation and gradient clipping;
- NTP/MTP loss configuration;
- AdamW or Kimi-style Muon/AdamW parameter groups;
- warmup and cosine scheduling;
- EMA and EMA evaluation;
- structured loss, routing, gradient and numerical diagnostics;
- progressive context stages;
- checkpoint/resume behavior;
- periodic next-token qualitative previews.

To start a real run from the CLI, remove `--validate-only`:

```bash
python -m scripts.train_kimi \
  --profile config/kimi_full_pipeline/low_gpu
```

This may download/tokenize data and allocate the configured model.

## Inference

Inference restores the architecture from `model.yaml`, loads a training
checkpoint and performs one prompt prefill followed by one cached decode step
per generated token:

```text
prompt -> KimiK3.prefill() -> HybridBackboneCache
       -> KimiK3.decode_step() -> next token -> repeat
```

KDA layers retain recurrent matrix and short-convolution state; MLA layers
retain compressed latent KV state. The full prefix is not recomputed.

```python
from data import load_tokenizer_from_data_yaml
from inference import (
    ModelLoadConfig,
    inference_autoregressive,
    load_generation_config,
    load_kimi_checkpoint,
)

tokenizer = load_tokenizer_from_data_yaml(profile.data)
loaded = load_kimi_checkpoint(
    profile.model,
    "checkpoints/model.pt",
    tokenizer=tokenizer,
    load_config=ModelLoadConfig(device="cuda", precision="fp16"),
)
generation = load_generation_config("config/inference/creative.yaml")
output = inference_autoregressive(
    loaded.model,
    "Once upon a time",
    tokenizer=tokenizer,
    generation_config=generation,
)
print(output.completion_text)
```

CLI:

```bash
python -m scripts.infer_kimi \
  --profile config/kimi_full_pipeline/low_gpu \
  --checkpoint checkpoints/model.pt \
  --inference-config config/inference/creative.yaml \
  --prompt "Once upon a time"
```

Sampling supports greedy decoding, temperature, top-k, top-p, repetition
penalty and deterministic seeds. See [`inference/README.md`](inference/README.md).

> [!WARNING]
> Only load checkpoints from trusted sources. PyTorch checkpoint
> deserialization is not a safe boundary for arbitrary files.

## Testing and CI

Tests are treated as architectural specifications. They check reference
equations, parameter wiring, causality, padding, gradients, BF16 behavior,
serialization and full-vs-cached equivalence—not only output shapes.

```bash
make check             # configuration + inference contracts
make test-config
make test-inference
make test-training
make test              # complete CPU-safe suite
```

The GitHub Actions workflow classifies changed paths:

| Change | CI lane |
|---|---|
| README/docs/community files only | Classification only; no full test suite |
| YAML/configuration | Profile parsers, cross-file contracts and CLI validation |
| Inference/cache code | Focused inference tests |
| Architecture, data, training, dependencies or broad tests | Full CPU suite plus CLI validation |
| Docker files | Container build and focused container smoke |

Workflows use read-only token permissions, concurrency cancellation and pinned
action revisions. CUDA checks skip safely when CUDA is unavailable.

The detailed invariant matrix is in [`docs/testing.md`](docs/testing.md).

## Docker

The supplied image is a reproducible CPU validation/inference environment. It
uses a multi-stage build, installs optional data dependencies and runs as a
non-root user.

```bash
docker compose run --rm validate
docker compose --profile test run --rm tests
docker compose --profile dev run --rm shell
```

Checkpoint inference:

```bash
KIMI_CHECKPOINT=checkpoints/model.pt \
KIMI_PROMPT="Once upon a time" \
docker compose --profile inference run --rm inference
```

The checkpoint and tokenizer-cache mounts are read-only. GPU execution needs a
CUDA-compatible PyTorch base image/runtime and is intentionally not implied by
the default CPU container.

## Repository layout

```text
.
├── config/kimi_full_pipeline/  # complete data/model/training profiles
├── configuration/              # strict YAML and profile resolution
├── data/                       # synthetic and Hugging Face LM pipelines
├── src/                        # KDA, MLA, MoE, AttnRes, vision, MTP, losses
├── training/                   # master trainer, optimizer, diagnostics, PCC
├── inference/                  # loading, sampling, prefill/decode and cache
├── scripts/                    # training, inference and validation CLIs
├── notebooks/                  # end-to-end Jupyter examples
├── tests/                      # CPU-safe behavioral and numerical tests
├── docs/                       # phase reports and public contracts
├── paper/                      # local technical-report reference
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## Documentation

Recommended entry points:

- [YAML training pipeline](docs/yaml_training_pipeline.md)
- [Kimi component map](src/kimi_components/README.md)
- [Master forward contract](docs/kimi_k3_forward_contract.md)
- [KDA implementation report](docs/kda_phase_report.md)
- [Gated MLA implementation report](docs/mla_phase_report.md)
- [Hybrid backbone report](docs/hybrid_backbone_phase_report.md)
- [Stable LatentMoE report](docs/stable_latent_moe_phase_report.md)
- [Attention Residuals report](docs/attention_residuals_phase_report.md)
- [MTP alignment and scope](docs/mtp_phase_report.md)
- [Training engine](docs/basic_training_phase_report.md)
- [Optimizer and diagnostics](docs/training_phase_2_optimizer_diagnostics_report.md)
- [Progressive Context Curriculum](docs/training_phase_3_progressive_context_curriculum_report.md)

## Project status

The architecture, pretraining engine, YAML control plane and cached inference
pipeline are implemented. The remaining public milestones are:

1. **Train a proof model — in progress.** Run and publish a small end-to-end
   checkpoint with reproducible curves and qualitative generations.
2. **Add simplified parallelism.** Introduce educational data parallel and
   model parallel execution without pretending to reproduce the production
   distributed stack.
3. **Integrate post-training.** SFT, RL/policy-optimization and multi-teacher
   distillation losses already exist under `src/loss/`; their datasets,
   rollout/teacher orchestration and checkpointed training pipeline remain to
   be connected.

Production serving kernels, paged attention, continuous batching, quantized
caches and speculative MTP decoding are also outside the current scope.

## Citation and license

GitHub exposes citation metadata from [`CITATION.cff`](CITATION.cff). Please
cite the original Kimi K3 technical report first:

```bibtex
@techreport{kimi_team_kimi_k3_2026,
  author        = {{Kimi Team}},
  title         = {Kimi K3: Open Frontier Intelligence},
  year          = {2026},
  eprint        = {2607.24653},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2607.24653},
  url           = {https://arxiv.org/abs/2607.24653}
}
```

If this implementation itself was useful, cite the software as well:

```bibtex
@software{reyes_kimi_k3_mini_2026,
  author  = {Pablo Reyes},
  title   = {Kimi-K3 Mini},
  year    = {2026},
  url     = {https://github.com/pablo-reyes8/kimi-k3},
  version = {0.0.1}
}
```

The project is released under the [MIT License](LICENSE). Generic Transformer,
data and training infrastructure was adapted from the author's MIT-licensed
DeepSeek-V4 Mini project; there are no runtime imports from that repository.
Kimi K3 itself and its technical report are the work of the Kimi Team.

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md),
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) and [`SECURITY.md`](SECURITY.md)
before opening a pull request or report.
