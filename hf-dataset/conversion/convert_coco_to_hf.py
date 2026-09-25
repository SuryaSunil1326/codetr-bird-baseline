"""Convert COCO JSON to HF Parquet dataset (streaming-compatible)."""
import argparse, json
from collections import defaultdict
from pathlib import Path
from datasets import Dataset, Features, Image as HFImage, Value, Sequence, ClassLabel

def convert(dataset_dir: Path, out_dir: Path, year: int = 2024):
    coco_path = dataset_dir / "annotations" / f"{year}_annotations.json"
    images_dir = dataset_dir / "images"
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco.get("categories", [{"id": 1, "name": "Bird"}])
    cat_names = [c["name"] for c in categories]
    cat_id_to_idx = {c["id"]: idx for idx, c in enumerate(categories)}

    anns = defaultdict(list)
    for a in coco.get("annotations", []):
        anns[a["image_id"]].append(a)

    def gen():
        for img in coco["images"]:
            img_path = images_dir / img["file_name"]
            if not img_path.exists():
                continue
            bboxes, cats_idx, areas, crowd, segmentations = [], [], [], [], []
            for a in anns.get(img["id"], []):
                bboxes.append(a["bbox"])
                cats_idx.append(cat_id_to_idx.get(a["category_id"], 0))
                areas.append(a.get("area", 0.0))
                crowd.append(a.get("iscrowd", 0))
                segmentations.append(a.get("segmentation", []))
            yield {
                "image_id": img["id"],
                "image": str(img_path),
                "file_name": img["file_name"],
                "height": img.get("height"),
                "width": img.get("width"),
                "objects": {"bbox": bboxes, "category": cats_idx, "area": areas, "iscrowd": crowd, "segmentation": segmentations},
            }

    features = Features({
        "image_id": Value("int64"),
        "image": HFImage(),
        "file_name": Value("string"),
        "height": Value("int32"),
        "width": Value("int32"),
        "objects": Sequence({
            "bbox": Sequence(Value("float32"), length=4),
            "category": ClassLabel(names=cat_names),
            "area": Value("float32"),
            "iscrowd": Value("int8"),
            "segmentation": Sequence(Sequence(Value("float32"))),
        }),
    })
    ds = Dataset.from_generator(gen, features=features)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out_dir / "data-00000-of-00001.parquet"))
    print(f"Wrote Parquet dataset to: {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--year", type=int, default=2024, help="Year of dataset (2024, 2025, 2026)")
    args = p.parse_args()
    convert(args.dataset_dir, args.output_dir, args.year)
