#!/usr/bin/env python3
"""Convert Chester Island JSON annotations to COCO format for SAM3/ultralytics."""
import csv
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# COCO conversion logic
# ---------------------------------------------------------------------------


def convert_json_to_coco(json_path: Path, output_path: Path, include_score: bool = False) -> None:
    """Convert Chester Island JSON to COCO format for SAM3/ultralytics.
    
    Args:
        json_path: Path to the input JSON file
        output_path: Path to the output COCO JSON file
        include_score: Whether to include the 'score' field in annotations
    """
    # Load JSON
    with open(json_path) as f:
        data = json.load(f)

    images = data["images"]
    detections = []
    categories = []
    category_map = {}

    # Build category map from species codes
    species_codes_file = json_path.parent / "JSon Color Defns.csv"
    if species_codes_file.exists():
        with open(species_codes_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                species_codes_file = row["Species Code"].strip()
                category_map[species_codes_file] = row["Desc"].strip()

    # Create categories (all are "Bird")
    bird_category_id = 1
    categories.append({
        "id": bird_category_id,
        "name": "Bird",
        "supercategory": "none"
    })

    # Convert images
    image_id = 1
    coco_images = []
    for img in images:
        file_name = img["file_name"]
        width = img["width"]
        height = img["height"]

        coco_images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })
        image_id += 1

    # Convert detections
    detection_id = 1
    coco_annotations = []
    for img in images:
        file_name = img["file_name"]
        # Find matching image in COCO
        coco_image_id = None
        for coco_img in coco_images:
            if coco_img["file_name"] == file_name:
                coco_image_id = coco_img["id"]
                break

        if coco_image_id is None:
            continue

        detections = img.get("detections", [])
        for det in detections:
            # COCO bbox format: [x, y, width, height]
            # But our data has: [x_min, y_min, x_max, y_max] (4 values)
            x_min, y_min, x_max, y_max = det["bbox"]
            width = x_max - x_min
            height = y_max - y_min

            # Convert to COCO format (0-indexed)
            x = x_min
            y = y_min

            # Build annotation without score by default
            # Ensure polygons are closed (first point = last point) per COCO spec
            seg = det.get("segmentation", [])
            closed_seg = []
            for poly in seg:
                if len(poly) >= 4 and (poly[0] != poly[-2] or poly[1] != poly[-1]):
                    closed_poly = poly + [poly[0], poly[1]]
                    closed_seg.append(closed_poly)
                else:
                    closed_seg.append(poly)
            
            ann = {
                "id": detection_id,
                "image_id": coco_image_id,
                "category_id": bird_category_id,
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": closed_seg
            }
            
            # Only include score when explicitly requested
            if include_score:
                ann["score"] = det.get("score", 1.0)
            
            coco_annotations.append(ann)
            detection_id += 1

    # Write COCO JSON (minimal structure)
    coco_data = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories
    }

    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"Converted {json_path.name} -> {output_path.name}")
    print(f"  Images: {len(coco_images)}")
    print(f"  Annotations: {len(coco_annotations)}")
    print(f"  Categories: {len(categories)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python json_to_coco.py <input_json> <output_coco> [--score]")
        return 1

    json_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    include_score = "--score" in sys.argv

    if not json_path.exists():
        print(f"Error: {json_path} does not exist")
        return 1

    convert_json_to_coco(json_path, output_path, include_score)
    return 0


if __name__ == "__main__":
    sys.exit(main())