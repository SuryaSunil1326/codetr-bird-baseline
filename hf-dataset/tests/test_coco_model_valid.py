#!/usr/bin/env python3
"""Validate COCO JSON is model-appropriate for 2024, 2025, 2026 datasets."""
import json
import pytest
from pathlib import Path

BASE_DIR = Path("/Users/jeffreyjoan1/Documents/git/Bird Detection ")

YEARS = [2024, 2025, 2026]


@pytest.mark.parametrize("year", YEARS)
def test_coco_model_appropriate(year):
    """Validate COCO JSON for a given year."""
    coco_file = BASE_DIR / f"Dataset/{year}/annotations/{year}_annotations.json"
    images_dir = BASE_DIR / f"Dataset/{year}/images"

    with open(coco_file) as f:
        coco = json.load(f)

    # 1. Required sections
    assert "categories" in coco, "Missing categories section"
    assert len(coco["categories"]) > 0, "Categories empty"
    cat_ids = {c["id"] for c in coco["categories"]}
    assert 1 in cat_ids, "Expected category id 1 (Bird)"

    assert "images" in coco, "Missing images section"
    assert "annotations" in coco, "Missing annotations section"

    # 2. Image field completeness + file existence + unique ids
    img_ids = set()
    for img in coco["images"]:
        assert "id" in img, f"Image missing 'id': {img}"
        assert "file_name" in img, f"Image missing 'file_name': {img}"
        assert "width" in img, f"Image missing 'width': {img}"
        assert "height" in img, f"Image missing 'height': {img}"
        assert isinstance(img["width"], (int, float)) and img["width"] > 0, f"Image width must be positive: {img['width']}"
        assert isinstance(img["height"], (int, float)) and img["height"] > 0, f"Image height must be positive: {img['height']}"
        
        img_ids.add(img["id"])
        
        img_path = images_dir / img["file_name"]
        assert img_path.exists(), f"Image file not found: {img_path}"

    assert len(img_ids) == len(coco["images"]), "Duplicate image ids"

    # 3. Annotation links + required fields
    ann_ids = set()
    for ann in coco["annotations"]:
        assert "id" in ann, f"Annotation missing 'id': {ann}"
        assert "image_id" in ann, f"Annotation missing 'image_id': {ann}"
        assert "category_id" in ann, f"Annotation missing 'category_id': {ann}"
        assert "bbox" in ann, f"Annotation missing 'bbox': {ann}"
        assert "area" in ann, f"Annotation missing 'area': {ann}"
        assert "iscrowd" in ann, f"Annotation missing 'iscrowd': {ann}"
        assert "segmentation" in ann, f"Annotation missing 'segmentation': {ann}"
        
        ann_ids.add(ann["id"])
        
        assert ann["image_id"] in img_ids, f"Annotation {ann['id']} links to missing image {ann['image_id']}"
        assert ann["category_id"] in cat_ids, f"Annotation {ann['id']} links to missing category {ann['category_id']}"

        # 4. Segmentation validity (polygons, bounds, closure)
        seg = ann["segmentation"]
        assert isinstance(seg, list), f"Annotation {ann['id']} segmentation not a list"
        assert len(seg) > 0, f"Annotation {ann['id']} segmentation is empty"
        
        img = next(i for i in coco["images"] if i["id"] == ann["image_id"])
        img_w, img_h = img["width"], img["height"]
        
        for poly_idx, poly in enumerate(seg):
            assert isinstance(poly, list), f"Annotation {ann['id']} polygon {poly_idx} not a list"
            assert len(poly) >= 6, f"Annotation {ann['id']} polygon {poly_idx} too short (need >=3 points)"
            assert len(poly) % 2 == 0, f"Annotation {ann['id']} polygon {poly_idx} odd coordinate count"
            
            # Coordinate bounds check
            for i in range(0, len(poly), 2):
                x, y = poly[i], poly[i+1]
                assert isinstance(x, (int, float)), f"Annotation {ann['id']} poly {poly_idx} x not numeric"
                assert isinstance(y, (int, float)), f"Annotation {ann['id']} poly {poly_idx} y not numeric"
                assert 0 <= x <= img_w, f"Annotation {ann['id']} poly {poly_idx} x={x} out of [0,{img_w}]"
                assert 0 <= y <= img_h, f"Annotation {ann['id']} poly {poly_idx} y={y} out of [0,{img_h}]"
            
            # Polygon closure
            assert poly[0] == poly[-2] and poly[1] == poly[-1], \
                f"Annotation {ann['id']} poly {poly_idx} not closed"

        # 5. Bbox format
        bbox = ann["bbox"]
        assert isinstance(bbox, list) and len(bbox) == 4, f"Annotation {ann['id']} bbox invalid: {bbox}"
        assert all(isinstance(v, (int, float)) for v in bbox), f"Annotation {ann['id']} bbox non-numeric"
        assert bbox[2] > 0 and bbox[3] > 0, f"Annotation {ann['id']} bbox w/h must be positive"
        
        # 6. Area matches bbox
        expected_area = bbox[2] * bbox[3]
        assert abs(ann["area"] - expected_area) < 0.01, f"Annotation {ann['id']} area mismatch"
        
        # 7. iscrowd valid
        assert ann["iscrowd"] in (0, 1), f"Annotation {ann['id']} iscrowd invalid"
        
        # 8. Score excluded by default
        assert "score" not in ann, f"Annotation {ann['id']} should not have score"

    # 9. Segmentation preserved
    first_ann = coco["annotations"][0]
    assert first_ann["segmentation"][0][0] > 0, "Segmentation coordinates should be positive"

    print(f"PASS: COCO format is model-appropriate for {year}.")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Categories: {len(coco['categories'])}")
    print(f"  Segmentation preserved: yes (first ann has {len(first_ann['segmentation'])} polygon(s))")
    print(f"  Score excluded by default: yes")


if __name__ == "__main__":
    for year in YEARS:
        test_coco_model_appropriate(year)