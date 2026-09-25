# codetr-bird-baseline

Co-DINO (ViT-Large, 5-scale) instance segmentation baseline for single-class bird detection, built on Co-DETR and MMDetection 2.25.3.

## Setup

    docker build -t codetr-bird .
    docker run --gpus all -it --rm --shm-size=8g -v "$PWD":/workspace codetr-bird
    python tools/smoke_test.py

Needs Docker, an NVIDIA driver and the NVIDIA container toolkit. The first build compiles mmcv-full from source (about 20 minutes). Run everything below inside the container.

## Data

COCO-format annotations and images go under `data/`, layout in `data/README.md`. `python tools/make_dummy_data.py` writes a tiny synthetic set there to check the pipeline (it will not overwrite existing annotations).

## Starting weights

Required before training. Download them as described in `checkpoints/README.md` (Sense-X Co-DINO ViT-L, COCO instance).

## Training

    python tools/train.py projects/configs/my_exps/co_dino_5scale_vit_large_bird_instance.py --work-dir work_dirs/run

Needs a large-memory GPU: the shipped config (AdamW, 7 images per GPU) does not fit in 6 GB even at batch size 1.

## Inference

    python tools/test.py projects/configs/my_exps/co_dino_5scale_vit_large_bird_instance.py work_dirs/run/latest.pth --eval bbox segm

## License

MIT for this repo's own code. mmdet/ is Apache-2.0 and Co-DETR is MIT, see NOTICE.
