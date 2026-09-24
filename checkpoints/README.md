Starting weights: Co-DINO-Inst, ViT-L, COCO instance segmentation (Sense-X Co-DETR release).

Download from https://huggingface.co/zongzhuofan/co-detr-vit-large-coco-instance (file `pytorch_model.pth`)
and save it here as `coco_dino_5scale_vit_large_coco_instance.pth`, the name the config expects:

    wget -O checkpoints/coco_dino_5scale_vit_large_coco_instance.pth \
      https://huggingface.co/zongzhuofan/co-detr-vit-large-coco-instance/resolve/main/pytorch_model.pth

The file is about 3 GB and is not tracked by git.
