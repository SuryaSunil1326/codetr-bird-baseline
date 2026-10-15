# Co-DINO Bird Detection Baseline - Agent Instructions

## Overview
Co-DINO (ViT-Large, 5-scale) instance segmentation for single-class bird detection. Built on Co-DETR + MMDetection 2.25.3.

## Environment
- **Must run inside Docker container** (Dockerfile compiles mmcv-full from source, ~20 min, 15 GB)
- Requires NVIDIA GPU, driver, and container toolkit
- Inside container: `python tools/smoke_test.py` verifies setup

## Key Commands

### Training
```bash
python tools/train.py projects/configs/my_exps/co_dino_5scale_vit_large_bird_instance.py --work-dir work_dirs/run
```
- Config expects 7 images/GPU; OOM? Add `--cfg-options data.samples_per_gpu=N` with smaller N
- Distributed: `./tools/dist_train.sh <config> <gpus> <workdir>`

### Inference
```bash
python tools/test.py projects/configs/my_exps/co_dino_5scale_vit_large_bird_instance.py work_dirs/run/latest.pth --eval bbox segm
```
- Distributed: `./tools/dist_test.sh <config> <checkpoint> <gpus>`

### Data Preparation
```bash
python tools/make_dummy_data.py  # writes synthetic COCO data to data/ (won't overwrite)
```
- Real data: COCO format under `data/annotations/{instances_train,instances_val}.json` and `data/images/{train,val}/`

### Starting Weights (required before training)
```bash
python -c "import urllib.request as u; u.urlretrieve('https://huggingface.co/zongzhuofan/co-detr-vit-large-coco-instance/resolve/main/pytorch_model.pth', 'checkpoints/coco_dino_5scale_vit_large_coco_instance.pth')"
```
- Expected sha256: `d21b1ddd0ac501b72d7d9d48390bcadbdf76e3195a976f7c6d0c93a6fb5046bf`
- Config references: `load_from = "checkpoints/coco_dino_5scale_vit_large_coco_instance.pth"`

## Project Structure
- `mmdet/` — Patched MMDetection 2.25.3 (see `mmdet/PATCHES.md` for changes: ViT backbone, SFP neck, RefineMaskHead, BARCrossEntropyLoss, layer-decay optimizer)
- `projects/configs/my_exps/co_dino_5scale_vit_large_bird_instance.py` — Main config (single-class bird, 512x512, 130 epochs, AdamW lr=5e-5)
- `tools/` — train.py, test.py, smoke_test.py, make_dummy_data.py, dist_train.sh, dist_test.sh
- `data/` — COCO annotations/images (gitignored except README)
- `checkpoints/` — Starting weights (gitignored except README)
- `work_dirs/` — Training outputs (gitignored)

## Config System
- Python-based MMDetection configs (not YAML)
- Base configs in `projects/configs/_base_/`
- Override via CLI: `--cfg-options key=value` (nested: `key="[a,b]"`, `key="[(a,b),(c,d)]"`)

## No Standard Tooling
- No lint, typecheck, pytest, or pre-commit configured
- This is a research/training repo; verification = smoke test + training convergence
- Code style: isort (line_length=79), yapf (pep8-based) per `setup.cfg`

## Important Notes
- `projects/__init__.py` imports models so Co-DETR components register (`from projects import *` in train/test)
- Single category: `bird` (class_id=1)
- Images tiled at 512x512
- ViT-Large backbone: 24 layers, embed_dim=1024, window attention

## Data Upload / Publishing (HF Hub)
- Raw COCO data goes in `data/annotations/{instances_train,instances_val}.json` + `data/images/{train,val}/`
- Convert to HF Parquet: `python hf-dataset/conversion/convert_coco_to_hf.py --dataset-dir data --output-dir hf-data --year 2024`
- Upload images (batched): `python hf-dataset/upload/upload_images_hf_batched.py --dataset-dir data/images --repo-id <org>/<repo> --branch V03 --clean-first`
- Upload full dataset folder: `python hf-dataset/upload/upload_dataset.py --dataset-dir hf-data --repo-id <org>/<repo> --branch V03`
- Target branch: `V03` (version tag for publish)
- Requires `huggingface_hub` and `datasets` packages, HF token via `huggingface-cli login`