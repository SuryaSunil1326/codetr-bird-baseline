import json
import math
import os
import random

from PIL import Image, ImageDraw

OUT = 'data'
SIZE = 512
SPLITS = {'train': 6, 'val': 3}


def ellipse_polygon(cx, cy, rx, ry, n=24):
    return [(cx + rx * math.cos(2 * math.pi * i / n),
             cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def polygon_area(pts):
    return 0.5 * abs(sum(x0 * y1 - x1 * y0
                         for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])))


if os.path.exists(f'{OUT}/annotations/instances_train.json'):
    raise SystemExit(f'{OUT}/annotations/instances_train.json already exists, not overwriting')

random.seed(0)
ann_id = 1
for split, n_images in SPLITS.items():
    os.makedirs(f'{OUT}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUT}/annotations', exist_ok=True)
    images, annotations = [], []
    for img_id in range(1, n_images + 1):
        img = Image.new('RGB', (SIZE, SIZE), (90, 120, 90))
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(3, 8)):
            cx, cy = random.randint(40, SIZE - 40), random.randint(40, SIZE - 40)
            rx, ry = random.randint(8, 24), random.randint(6, 16)
            pts = ellipse_polygon(cx, cy, rx, ry)
            draw.polygon(pts, fill=(235, 235, 235))
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            annotations.append({
                'id': ann_id, 'image_id': img_id, 'category_id': 1, 'iscrowd': 0,
                'bbox': [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                'area': polygon_area(pts),
                'segmentation': [[c for p in pts for c in p]]})
            ann_id += 1
        name = f'{split}_{img_id:03d}.jpg'
        img.save(f'{OUT}/images/{split}/{name}')
        images.append({'id': img_id, 'file_name': name, 'width': SIZE, 'height': SIZE})
    with open(f'{OUT}/annotations/instances_{split}.json', 'w') as f:
        json.dump({'images': images, 'annotations': annotations,
                   'categories': [{'id': 1, 'name': 'bird'}]}, f)
    print(split, len(images), 'images')
