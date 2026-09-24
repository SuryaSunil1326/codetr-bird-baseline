# mmdet patches

This directory is MMDetection v2.25.3 (https://github.com/open-mmlab/mmdetection) with the
modifications made by Sense-X/Co-DETR (https://github.com/Sense-X/Co-DETR, commit
26653521e47157fdd819e764b4bb949057ea9f3b). The patches below come from that repository.

Listed by diffing against the upstream v2.25.3 tarball; paths are relative to `mmdet/`.

## New files

| File | What it adds |
|---|---|
| `models/backbones/vit.py` | `ViT` backbone (rotary position embedding, SwiGLU MLP, windowed attention blocks) |
| `models/necks/sfp.py` | `SFP` neck, a Simple Feature Pyramid over the single-scale ViT output |
| `models/roi_heads/mask_heads/refine_mask_head.py` | `RefineMaskHead` and `SimpleRefineMaskHead` mask heads (multi-stage mask refinement) |

## Modified files

| File | Change |
|---|---|
| `models/backbones/__init__.py` | registers `ViT` |
| `models/necks/__init__.py` | registers `SFP` |
| `models/roi_heads/mask_heads/__init__.py` | exports `SimpleRefineMaskHead` |
| `models/losses/__init__.py` | exports `BARCrossEntropyLoss` |
| `models/losses/cross_entropy_loss.py` | adds `generate_block_target` (boundary/foreground block targets from a mask) and `BARCrossEntropyLoss` (boundary-aware mask loss) |
| `models/roi_heads/mask_heads/maskiou_head.py` | `MaskIoUHead` gets `score_use_sigmoid` and `norm_cfg` options (ConvModule convs) |
| `models/utils/transformer.py` | `DetrTransformerEncoder` gets `with_cp` activation checkpointing (fairscale `checkpoint_wrapper`); drops the post-norm `forward` override |
| `models/dense_heads/detr_head.py` | removes the asserts that loss weights equal assigner cost weights |
| `core/optimizers/layer_decay_optimizer_constructor.py` | adds `get_layer_id_for_vit` for layer-wise learning-rate decay on ViT |
| `core/evaluation/class_names.py`, `core/evaluation/__init__.py` | adds `DatasetEnum` and LVIS class names |
| `apis/inference.py` | `init_detector` takes a `dataset` (`DatasetEnum`) for default class names; imports `projects` |
| `apis/test.py`, `apis/train.py` | `from projects import *` so the Co-DETR models register |
| `datasets/builder.py` | drops `ann_file`, `img_prefix`, `filter_empty_gt` before building `MultiImageMixDataset` |
| `datasets/lvis.py` | `LVISResults(..., max_dets=1000)` |

## Differences from Sense-X/Co-DETR

Compared against the same commit:

- `datasets/coco.py`: one added `print("[DEBUG] ...")` line inside the annotation loop. Not from Co-DETR.
- `projects/models/swin_transformer.py` is not included and `projects/models/__init__.py` does not
  import it (the ViT config does not use Swin, and it depends on `mmcv_custom/`, which is also not included).
- `tools/train.py`, `tools/test.py` and the other `projects/models/` files are identical to upstream.
