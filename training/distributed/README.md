# Kimi distributed subsystem

This package is the compatibility boundary between the existing Kimi model
and `torch.distributed`.

## Supported composition

```text
world rank
   └── DP coordinate × TP coordinate × EP coordinate

DP: none | DDP | FSDP
TP: tied vocabulary + complete KDA/MLA heads
EP: contiguous routed-expert ownership + variable all_to_all
```

`world_size` must equal `dp_size * tp_size * ep_size`. Tensor-parallel ranks
within one TP group receive the same data. EP ranks receive distinct sampler
shards; replicated parameters are averaged across EP while routed expert
parameters stay on their owners.

The first Kimi TP cache layout is deliberate:

- KDA recurrent matrices and three ShortConv histories are local by complete
  head;
- MLA query/key/value heads are local;
- the compressed MLA latent cache is replicated;
- AttnRes state, norms, vision and non-routed MoE paths remain replicated.

## Boundaries

- PP and cross-device KDA context parallelism are validation-only and must have
  size one.
- FSDP uses the installed PyTorch implementation and requires an accelerator
  in PyTorch builds that reject CPU FSDP.
- EMA with FSDP is rejected because no hidden full-model replica is allowed.
- Same-topology distributed resume is supported. Elastic resharding and a
  consolidated TP/EP export are not claimed.
- This is not MoonEP, Mooncake offload, Pipeline ZeRO, or a custom production
  kernel stack.

All normal entry points flow through `train_kimiK3`; do not build a second
distributed model class or training loop.
