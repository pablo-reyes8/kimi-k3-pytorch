# MoonViT migration inventory

This inventory was completed before implementing `src/vision`.  The donor
repository is `multiscale-vision-transformers/model_zoo`; it is MIT licensed.
The new code must not import that repository at runtime.

## Decision

MoonViT V2 Mini uses a flat, global Vision Transformer as its canonical
backbone. This is the closest implementation to the public Kimi K3 report:
global self-attention, patch size 14, 27 blocks in the reported large model,
12 heads, RMSNorm, no CLS token, and bias-free linear/attention layers.

Two explicit research variants are also in scope: `HierarchicalMoonViT`
(global attention plus progressive token pooling) and `SwinMoonViT` (shifted
window attention plus patch merging). They remain separate from the canonical
path. MaxViT and VOLO are inventory-only: MBConv, multi-axis attention, and
Outlook Attention are not implemented.

## Standard ViT

- Source: `model/patch_embedding.py`, `model/attention_blocks.py`,
  `model/vision_transformer.py`.
- Patch embedding: non-overlapping `Conv2d(kernel=stride=patch_size)`;
  `[B,C,H,W] -> [B,N,D]`; rectangular and dynamically sized inputs work when
  divisible by the patch size.
- Position: learned absolute patch grid plus a mandatory learned CLS position;
  bicubic interpolation supports grids different from the configured base.
- Attention/block: fused QKV global MHSA, pre-LayerNorm residual block, GELU
  two-layer MLP, dropout and stochastic depth.
- Output/head: CLS and patch features are available internally, but public
  forward returns classification logits.
- Gaps for MoonViT: mandatory CLS, LayerNorm, classifier coupling, projection
  biases, no visual padding mask, and no diagnostic hidden states/attentions.
- Tests: basic shape/finite checks and a short CPU classification train/eval
  test.  Useful starting points, but insufficient for the project standard.
- Reuse: global-attention topology, conv patchification, interpolated learned
  positions, pre-norm block layout, and per-layer drop-path schedule.

## SwinViT

- Source: `model/Vit_embeddings.py`, `window_partition.py`,
  `swin_attention.py`, `swin_block.py`, `patch_merging.py`,
  `backbone_block.py`, `swin_vision_transformer.py`.
- Patch embedding: conv patchification to channel-last maps, optional padding,
  optional token view, and LayerNorm.
- Position/attention: relative position-bias tables inside local windows;
  alternating regular/shifted windows with additive attention masks.
- Hierarchy: 2x2 `PatchMerging` concatenates four spatial positions, normalizes,
  then projects `4C -> 2C`; padding permits odd grids.
- Output/head: masked global average after four stages and a classifier.
- Variable resolution: patch and window padding are supported.  Stage window
  sizes are clipped from the configured resolution, and padded patches are
  excluded from final pooling.
- Tests: patch shapes, exact prepare/restore roundtrip, patch merging, and one
  small end-to-end shape test.
- Reuse: careful grid bookkeeping and the exact 2x2 spatial concatenation idea
  informed MoonViT's *post-encoder* pixel packing.  Window attention and stage
  merging are deliberately not migrated.

## HierarchicalViT

- Source: `model/patch_embedding.py`, `attention_blocks.py`,
  `transformer_pooling.py`, `hierarchical_vit.py`.
- Patch embedding: strict-divisibility conv patchification followed by optional
  token normalization.
- Position/attention: learned absolute position table per stage; global MHSA
  with several supported mask ranks; pre-LayerNorm GELU blocks.
- Hierarchy: depthwise 3x3 stride-2 convolution then pointwise projection.
- Output/head: mean token pooling, LayerNorm, linear classifier.
- Variable resolution: patchification itself is dynamic, but the model asserts
  fixed stage grids and does not interpolate its per-stage positions.
- Tests: attention mask semantics, mask ranks, patch/pool shapes, classifier
  output.
- Reuse: explicit mask-shape normalization and token/grid assertions.  Its
  progressive pooling and stage-specific positions are not MoonViT.

## MaxViT

- Source: `model/MaxViT.py`, `MaxViTStage.py`, `MaxViT_block.py`,
  `attention.py`, `local_attention.py`, `window_partition.py`,
  `grid_partition.py`, `MBConv.py`, stem/downsample helpers.
- Input/stem: convolutional stem rather than ViT-style patch embedding.
- Attention: each block uses MBConv, local window attention, global
  modulo-grid attention, then a pre-LayerNorm MLP.
- Hierarchy: explicit convolution/pooling between stages; NCHW internally and
  BHWC for attention.
- Position: no absolute patch position table in the inspected implementation.
- Output/head: adaptive average pool and linear classifier.
- Variable resolution: partition utilities require exact divisibility by their
  window/grid size and validate rank/shape rigorously.
- Tests: exact partition roundtrips, invalid-input errors, forward/backward for
  attention, MBConv, blocks and full model, plus CPU training helpers.
- Dependencies: its project declares `timm` and `einops`, but the MoonViT
  migration takes no dependency on either.
- Reuse: strong public-boundary validation and exact inverse-property tests.
  MBConv and multi-axis partition attention are excluded from the canonical
  architecture.

## VOLO

- Source: `model/embeddings.py`, `outlook.py`, `attention.py`,
  `volo_stage.py`, `pooling_volo_blocks.py`, `cls_attention.py`, `VOLO.py`.
- Patch embedding: conv patchification to BHWC, optional padding and token
  view. `PosEmbed2D` performs bicubic learned-position interpolation.
- Attention: early Outlook Attention dynamically aggregates local unfolded
  neighborhoods; later stages may use global Transformer blocks.
- Hierarchy: optional channel-last convolutional or token-space pooling.
- Output/head: mean, CLS, or class-attention pooling followed by classifier.
- Variable resolution: patch padding and positional interpolation are
  supported; pyramid grids are propagated explicitly.
- Tests: divisible/non-divisible patch cases, positional interpolation, local
  attention/block gradients, pyramid variants, classifier variants, and CPU
  training.
- Reuse: positional interpolation behavior, explicit spatial metadata, and
  stochastic-depth testing patterns.  Outlook Attention and classifier
  pooling are not MoonViT mechanisms.

## Training and data assessment

All five projects train image classifiers and duplicate AMP, checkpoint,
scheduler, metric, and epoch-loop utilities.  The Kimi repository already has
tested, task-agnostic equivalents, so none are copied.  CIFAR-100 loaders are
also intentionally omitted.  Vision optimization is verified with a tiny,
deterministic synthetic-shape task owned by the tests; no network download or
production classifier head is introduced.

## Migration boundaries

- Migrated conceptually: strict conv patchification, global pre-norm ViT,
  learned-position interpolation, stochastic-depth scheduling, explicit grid
  metadata, masks, and exact 2x2 packing.
- Reimplemented for MoonViT: RMSNorm support, independently configurable
  biases, no-CLS default, structured encoder outputs, optional attention
  diagnostics, visual masks, pixel packing, and multimodal projector.
- Implemented only in named ablations: Hierarchical token pooling and Swin
  shifted-window attention/patch merging.
- Not migrated: classification heads, dataset code, training duplication,
  MaxViT multi-axis attention, VOLO Outlook Attention, MBConv, or donor runtime
  imports.
