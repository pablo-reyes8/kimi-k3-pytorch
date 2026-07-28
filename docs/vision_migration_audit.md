# Vision migration audit

## Result

The repository now provides three independent visual encoder profiles:

1. `MoonViTEncoder`: canonical flat global ViT.
2. `HierarchicalMoonViTEncoder`: global-attention ablation with PiT-style
   spatial pooling between stages.
3. `SwinMoonViTEncoder`: shifted-window ablation with relative position bias
   and 2x2 patch merging.

MaxViT and VOLO were inspected but no production modules were copied.

## MoonViT invariants

- Patch size 14 is used by every MoonViT configuration.
- The canonical defaults use 12 heads, RMSNorm, no CLS token, no classifier,
  and no biases in patch, QKV, output, or MLP projections.
- The official report's 27 layers and approximately 401M parameters describe
  the full model, not the CPU-friendly proxy presets in this repository.
- Absolute learned positions are a provisional mechanism because the report
  does not identify MoonViT-V2's exact positional method. They can be disabled
  and are interpolated bicubically for dynamic rectangular grids.
- Spatial token packing reduces an even `H x W` grid to
  `H/2 x W/2`, concatenating channels in TL, TR, BL, BR order.
- The projector is isolated from the encoder and maps packed visual width to
  the language-model width without introducing a classifier.

## Intentional differences between variants

| Profile | Attention | Resolution changes | Position |
|---|---|---|---|
| MoonViT | Global at every block | Only final 2x2 token packing | Optional learned absolute |
| Hierarchical | Global within each stage | Depthwise stride-2 pool + projection | Optional learned absolute per stage |
| Swin | Regular/shifted local windows | 2x2 patch merging | Window-relative bias |

The hierarchical and Swin profiles are research ablations, not claims about
the official Kimi K3 visual backbone.

## Donor mapping

| Donor area | Destination | Status |
|---|---|---|
| ViT patch embedding | `src/vision/patch_embedding.py` | Reimplemented and hardened |
| ViT global attention/block | `vision_attention.py`, `vision_block.py` | Reimplemented for masks/RMSNorm/bias policy |
| ViT learned position interpolation | `positional_embedding.py` | Reimplemented |
| Hierarchical pooling | `hierarchical_encoder.py` | Adapted as named ablation |
| Swin windows/merging | `swin_encoder.py` | Adapted as named ablation |
| MaxViT | — | Inspected, not migrated |
| VOLO | — | Inspected, not migrated |
| Classification heads/data/training | — | Intentionally omitted |

The donor repository is MIT licensed. The implementation is a clean
adaptation with no runtime imports from `multiscale-vision-transformers`.

## Verification

Tests live in `tests/MoonViT`. They cover reference equations, exact spatial
ordering, configuration failures, structural bias policy, interpolation,
masks, shifted-window isolation, dynamic and rectangular resolution,
diagnostic outputs, batch independence, all-parameter gradients, BF16,
state-dict roundtrips, visual-path composition, and a download-free synthetic
spatial overfit.

