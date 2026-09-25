Starting weights: Co-DINO-Inst, ViT-L, COCO instance segmentation (Sense-X Co-DETR release).

Source: https://huggingface.co/zongzhuofan/co-detr-vit-large-coco-instance (file `pytorch_model.pth`, about 3 GB).
Save it here as `coco_dino_5scale_vit_large_coco_instance.pth`, the name the config expects. From the repo root, inside the container:

    python -c "import urllib.request as u; u.urlretrieve('https://huggingface.co/zongzhuofan/co-detr-vit-large-coco-instance/resolve/main/pytorch_model.pth', 'checkpoints/coco_dino_5scale_vit_large_coco_instance.pth')"
    sha256sum checkpoints/coco_dino_5scale_vit_large_coco_instance.pth

Expected sha256: d21b1ddd0ac501b72d7d9d48390bcadbdf76e3195a976f7c6d0c93a6fb5046bf

The file is not tracked by git.
