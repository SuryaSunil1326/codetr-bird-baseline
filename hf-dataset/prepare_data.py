#!/usr/bin/env python3
"""Prepare raw Chester Island data for HF upload.

Converts raw data/Publish V03/{year}/ into the standardized data/{year}/
structure that matches the HF Hub branch layout:

    data/{year}/
    ├── Images/          (jpg, jgw, mrx + CSVs)
    ├── annotations/{year}_annotations.json   (COCO format)
    └── parquet/data-00000-of-00001.parquet   (HF streaming)

Usage:
    python hf-dataset/prepare_data.py                          # all years
    python hf-dataset/prepare_data.py --year 2024              # single year
    python hf-dataset/prepare_data.py --year 2024 --force      # overwrite
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Allow importing from the conversion/ directory
sys.path.insert(0, str(Path(__file__).parent / "conversion"))

from json_to_coco import convert_json_to_coco


# Map of raw data directories to year labels
YEAR_DIRS = {
    2024: "20240518 - Chester Island 10k",
    2025: "20250520 - Chester Island 10K",
    2026: "20260519 - Chester Island 10k",
}


def prepare_year(raw_dir: Path, data_dir: Path, year: int, force: bool = False):
    """Convert one year of Chester Island data to the standardized format."""
    year_dir = data_dir / str(year)
    images_dir = year_dir / "Images"
    annotations_dir = year_dir / "annotations"
    parquet_dir = year_dir / "parquet"

    # Check if already prepared
    coco_path = annotations_dir / f"{year}_annotations.json"
    if coco_path.exists() and not force:
        print(f"[{year}] Already prepared (found {coco_path}). Skipping.")
        return

    # Find the Chester Island JSON file
    json_files = list(raw_dir.glob("*Annotated.json"))
    if not json_files:
        print(f"[{year}] ERROR: No *Annotated.json found in {raw_dir}")
        return
    chester_json = json_files[0]

    # Find the species CSV
    species_csv = raw_dir / "JSon Color Defns.csv"
    if not species_csv.exists():
        # Try alternate location (some years have it in Images/)
        species_csv = raw_dir / "Images" / "JSon Color Defns.csv"

    # Find AI Class Defns CSV (may be in raw_dir or Images/)
    ai_class_csv = raw_dir / "AI Class Defns.csv"
    if not ai_class_csv.exists():
        ai_class_csv = raw_dir / "Images" / "AI Class Defns.csv"

    print(f"\n{'='*60}")
    print(f"Preparing year: {year}")
    print(f"  Raw dir:       {raw_dir}")
    print(f"  Chester JSON:  {chester_json.name}")
    print(f"  Output:        {year_dir}")
    print(f"{'='*60}")

    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Convert Chester Island JSON -> COCO ---
    print(f"\n[1/4] Converting Chester Island JSON -> COCO...")
    convert_json_to_coco(chester_json, coco_path, include_score=False)
    print(f"  -> {coco_path}")

    # --- Step 2: Copy images and sidecar files ---
    print(f"\n[2/4] Copying images and sidecar files...")
    raw_images_dir = raw_dir / "Images"
    if raw_images_dir.exists():
        image_files = [
            f for f in raw_images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".jgw", ".jgwx", ".mrx")
        ]
        copied = 0
        for f in image_files:
            dst = images_dir / f.name
            if force or not dst.exists():
                shutil.copy2(f, dst)
                copied += 1
        print(f"  Copied {copied} files to {images_dir}")
    else:
        print(f"  WARNING: No Images/ directory found in {raw_dir}")

    # --- Step 3: Copy CSV metadata files ---
    print(f"\n[3/4] Copying CSV metadata files...")
    csvs_copied = 0
    for csv_path in [species_csv, ai_class_csv]:
        if csv_path and csv_path.exists():
            dst = images_dir / csv_path.name
            if force or not dst.exists():
                shutil.copy2(csv_path, dst)
                csvs_copied += 1
                print(f"  -> {csv_path.name}")
    print(f"  Copied {csvs_copied} CSV files")

    # --- Step 4: Generate HF Parquet ---
    print(f"\n[4/4] Generating HF Parquet dataset...")
    try:
        from datasets import Dataset, Features, Image as HFImage, Value, Sequence, ClassLabel
        from collections import defaultdict

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
                    "objects": {
                        "bbox": bboxes,
                        "category": cats_idx,
                        "area": areas,
                        "iscrowd": crowd,
                        "segmentation": segmentations,
                    },
                }

        features = Features({
            "image_id": Value("int64"),
            "image": HFImage(decode=False),
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
        parquet_path = parquet_dir / "data-00000-of-00001.parquet"
        ds.to_parquet(str(parquet_path))
        print(f"  -> {parquet_path}")
        print(f"  Rows: {len(ds)}")

    except ImportError:
        print("  WARNING: `datasets` package not installed. Skipping parquet generation.")
        print("  Install with: pip install datasets")

    print(f"\n[{year}] Done!")


def main():
    parser = argparse.ArgumentParser(description="Prepare Chester Island data for HF upload")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/Publish V03"),
        help="Path to raw Publish V03 directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Output data directory",
    )
    parser.add_argument(
        "--year",
        type=int,
        choices=list(YEAR_DIRS.keys()),
        help="Prepare a single year (default: all years)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prepared data",
    )
    args = parser.parse_args()

    if not args.raw_dir.exists():
        print(f"ERROR: Raw directory not found: {args.raw_dir}")
        sys.exit(1)

    years = [args.year] if args.year else list(YEAR_DIRS.keys())

    for year in years:
        raw_dir = args.raw_dir / YEAR_DIRS[year]
        if not raw_dir.exists():
            print(f"WARNING: Raw directory for {year} not found: {raw_dir}")
            continue
        prepare_year(raw_dir, args.data_dir, year, force=args.force)

    print(f"\n{'='*60}")
    print("All years prepared!")
    print(f"Data is ready in: {args.data_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
