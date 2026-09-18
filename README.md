# codetr-bird-baseline

Single-class ("bird") CoDETR fine-tune baseline for student projects. Co-DINO 5-scale with ViT-Large backbone, instance segmentation head.

## Setup

    pip install -v -e .
    pip install -r requirements/runtime.txt

requires torch + torchvision installed separately (matching your CUDA version), and mmcv-full (see requirements/mminstall.txt).

## Data

Place COCO-format annotations and images under `data/`. See `data/README.md` for the expected layout.

## Pretrained weights

Download `coco_dino_5scale_vit_large_coco_instance.pth` from the Sense-X Co-DETR release and place it in `checkpoints/`.

## Train

    python tools/train.py projects/configs/my_exps/co_dino_5scale_vit_large_bird_instance.py --work-dir work_dirs/run

## Test

    python tools/test.py projects/configs/my_exps/co_dino_5scale_vit_large_bird_instance.py work_dirs/run/latest.pth --eval bbox segm
