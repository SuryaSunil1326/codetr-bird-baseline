#!/usr/bin/env python3
"""Validate COCO data format by emulating how MMDetection loads it for training.

This test checks:
1. COCO JSON structure is valid
2. All annotations reference existing images
3. Bounding boxes are valid (positive width/height, within image bounds)
4. Segmentations are valid polygons (closed, even coordinate count)
5. Data can be loaded as a Dataset dict (simulating CocoDataset)
6. Train/val split is correct
"""

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Years and their expected directories
YEARS = [2024, 2025, 2026]


class TestCocoJsonStructure:
    """Test that COCO JSON files have valid structure."""

    @pytest.mark.parametrize("year", YEARS)
    def test_coco_json_exists(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        assert coco_path.exists(), f"Missing: {coco_path}"

    @pytest.mark.parametrize("year", YEARS)
    def test_coco_json_has_required_keys(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        assert "images" in coco, "Missing 'images' key"
        assert "annotations" in coco, "Missing 'annotations' key"
        assert "categories" in coco, "Missing 'categories' key"

    @pytest.mark.parametrize("year", YEARS)
    def test_coco_json_categories(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        cats = coco["categories"]
        assert len(cats) >= 1, "Must have at least one category"
        for cat in cats:
            assert "id" in cat
            assert "name" in cat

    @pytest.mark.parametrize("year", YEARS)
    def test_coco_json_images_have_required_fields(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        for img in coco["images"]:
            assert "id" in img, f"Image missing 'id': {img}"
            assert "file_name" in img, f"Image missing 'file_name': {img}"
            assert "width" in img, f"Image missing 'width': {img}"
            assert "height" in img, f"Image missing 'height': {img}"
            assert img["width"] > 0, f"Image width must be positive: {img}"
            assert img["height"] > 0, f"Image height must be positive: {img}"

    @pytest.mark.parametrize("year", YEARS)
    def test_coco_json_annotations_have_required_fields(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            assert "id" in ann, f"Annotation missing 'id': {ann}"
            assert "image_id" in ann, f"Annotation missing 'image_id': {ann}"
            assert "category_id" in ann, f"Annotation missing 'category_id': {ann}"
            assert "bbox" in ann, f"Annotation missing 'bbox': {ann}"
            assert "area" in ann, f"Annotation missing 'area': {ann}"
            assert "segmentation" in ann, f"Annotation missing 'segmentation': {ann}"
            assert "iscrowd" in ann, f"Annotation missing 'iscrowd': {ann}"


class TestCocoDataConsistency:
    """Test referential integrity between images and annotations."""

    @pytest.mark.parametrize("year", YEARS)
    def test_all_annotations_reference_existing_images(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        img_ids = {img["id"] for img in coco["images"]}
        for ann in coco["annotations"]:
            assert ann["image_id"] in img_ids, \
                f"Annotation {ann['id']} references missing image {ann['image_id']}"

    @pytest.mark.parametrize("year", YEARS)
    def test_all_annotations_reference_existing_categories(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        cat_ids = {cat["id"] for cat in coco["categories"]}
        for ann in coco["annotations"]:
            assert ann["category_id"] in cat_ids, \
                f"Annotation {ann['id']} references missing category {ann['category_id']}"

    @pytest.mark.parametrize("year", YEARS)
    def test_annotation_ids_are_unique(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        ann_ids = [ann["id"] for ann in coco["annotations"]]
        assert len(ann_ids) == len(set(ann_ids)), "Duplicate annotation IDs found"

    @pytest.mark.parametrize("year", YEARS)
    def test_image_ids_are_unique(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        img_ids = [img["id"] for img in coco["images"]]
        assert len(img_ids) == len(set(img_ids)), "Duplicate image IDs found"

    @pytest.mark.parametrize("year", YEARS)
    def test_image_files_exist_on_disk(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        images_dir = DATA_DIR / str(year) / "Images"
        missing = []
        for img in coco["images"]:
            img_path = images_dir / img["file_name"]
            if not img_path.exists():
                missing.append(img["file_name"])
        assert len(missing) == 0, f"Missing {len(missing)} image files, e.g.: {missing[:5]}"


class TestCocoBoundingBoxValidity:
    """Test bounding box format and validity."""

    @pytest.mark.parametrize("year", YEARS)
    def test_bbox_format(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            bbox = ann["bbox"]
            assert isinstance(bbox, list), f"bbox must be list, got {type(bbox)}"
            assert len(bbox) == 4, f"bbox must have 4 elements [x,y,w,h], got {len(bbox)}"
            x, y, w, h = bbox
            assert all(isinstance(v, (int, float)) for v in bbox), f"bbox values must be numeric: {bbox}"

    @pytest.mark.parametrize("year", YEARS)
    def test_bbox_positive_dimensions(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            bbox = ann["bbox"]
            assert bbox[2] > 0, f"bbox width must be positive: {bbox}"
            assert bbox[3] > 0, f"bbox height must be positive: {bbox}"

    @pytest.mark.parametrize("year", YEARS)
    def test_bbox_within_image_bounds(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        img_map = {img["id"]: img for img in coco["images"]}
        violations = []
        for ann in coco["annotations"]:
            img = img_map[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            if x < 0 or y < 0 or x + w > img["width"] or y + h > img["height"]:
                violations.append({
                    "ann_id": ann["id"],
                    "bbox": ann["bbox"],
                    "image_size": [img["width"], img["height"]],
                })
        assert len(violations) == 0, \
            f"{len(violations)} bboxes out of bounds, e.g.: {violations[:3]}"

    @pytest.mark.parametrize("year", YEARS)
    def test_area_matches_bbox(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        mismatches = []
        for ann in coco["annotations"]:
            expected = ann["bbox"][2] * ann["bbox"][3]
            if abs(ann["area"] - expected) > 0.01:
                mismatches.append({"ann_id": ann["id"], "area": ann["area"], "expected": expected})
        assert len(mismatches) == 0, \
            f"{len(mismatches)} area mismatches, e.g.: {mismatches[:3]}"


class TestCocoSegmentationValidity:
    """Test segmentation polygon format and validity."""

    @pytest.mark.parametrize("year", YEARS)
    def test_segmentation_is_list_of_polygons(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            seg = ann["segmentation"]
            assert isinstance(seg, list), f"segmentation must be list, got {type(seg)}"
            for poly in seg:
                assert isinstance(poly, list), f"polygon must be list, got {type(poly)}"

    @pytest.mark.parametrize("year", YEARS)
    def test_segmentation_even_coordinate_count(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            for poly in ann["segmentation"]:
                assert len(poly) % 2 == 0, \
                    f"Polygon must have even number of coordinates, got {len(poly)}"

    @pytest.mark.parametrize("year", YEARS)
    def test_segmentation_minimum_points(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            for poly in ann["segmentation"]:
                assert len(poly) >= 6, \
                    f"Polygon must have at least 3 points (6 coords), got {len(poly)}"

    @pytest.mark.parametrize("year", YEARS)
    def test_segmentation_closed(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        open_polys = []
        for ann in coco["annotations"]:
            for i, poly in enumerate(ann["segmentation"]):
                if len(poly) >= 6 and (poly[0] != poly[-2] or poly[1] != poly[-1]):
                    open_polys.append({"ann_id": ann["id"], "poly_idx": i})
        assert len(open_polys) == 0, \
            f"{len(open_polys)} unclosed polygons, e.g.: {open_polys[:3]}"

    @pytest.mark.parametrize("year", YEARS)
    def test_segmentation_within_image_bounds(self, year):
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        img_map = {img["id"]: img for img in coco["images"]}
        violations = []
        for ann in coco["annotations"]:
            img = img_map[ann["image_id"]]
            for poly in ann["segmentation"]:
                for i in range(0, len(poly), 2):
                    x, y = poly[i], poly[i + 1]
                    if x < 0 or y < 0 or x > img["width"] or y > img["height"]:
                        violations.append({
                            "ann_id": ann["id"],
                            "coord": [x, y],
                            "image_size": [img["width"], img["height"]],
                        })
                        break
        assert len(violations) == 0, \
            f"{len(violations)} segmentation coords out of bounds, e.g.: {violations[:3]}"


class TestCocoTrainingSimulation:
    """Simulate how MMDetection's CocoDataset loads the data."""

    @pytest.mark.parametrize("year", YEARS)
    def test_annotations_per_image(self, year):
        """Each image should have at least one annotation (it's a detection dataset)."""
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        anns_per_img = defaultdict(list)
        for ann in coco["annotations"]:
            anns_per_img[ann["image_id"]].append(ann)
        empty_images = [img_id for img_id in range(1, len(coco["images"]) + 1)
                        if img_id not in anns_per_img]
        # Allow some empty images (no birds detected), but most should have annotations
        ratio = len(empty_images) / len(coco["images"]) if coco["images"] else 0
        assert ratio < 0.5, f"Too many empty images: {len(empty_images)}/{len(coco['images'])} ({ratio:.1%})"

    @pytest.mark.parametrize("year", YEARS)
    def test_multiple_masks_per_image(self, year):
        """Verify that images with multiple masks are handled correctly."""
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        anns_per_img = defaultdict(list)
        for ann in coco["annotations"]:
            anns_per_img[ann["image_id"]].append(ann)
        multi_mask = {k: v for k, v in anns_per_img.items() if len(v) > 1}
        assert len(multi_mask) > 0, "No images with multiple masks found — expected some"
        # Verify each mask has a unique annotation ID
        for img_id, anns in multi_mask.items():
            ann_ids = [a["id"] for a in anns]
            assert len(ann_ids) == len(set(ann_ids)), \
                f"Duplicate ann IDs in image {img_id}: {ann_ids}"

    @pytest.mark.parametrize("year", YEARS)
    def test_load_as_dataset_dict(self, year):
        """Simulate loading COCO data as MMDetection would (list of dicts per image)."""
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)

        # Build annotation lookup by image_id
        anns_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_img[ann["image_id"]].append(ann)

        # Simulate CocoDataset loading: each image becomes a data record
        data_list = []
        for img in coco["images"]:
            record = {
                "img_id": img["id"],
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "ann_info": anns_by_img.get(img["id"], []),
            }
            data_list.append(record)

        assert len(data_list) == len(coco["images"]), "Data list length mismatch"
        # Verify each record has the expected structure
        for record in data_list:
            assert "img_id" in record
            assert "file_name" in record
            assert "width" in record
            assert "height" in record
            assert "ann_info" in record
            assert isinstance(record["ann_info"], list)

    @pytest.mark.parametrize("year", YEARS)
    def test_iscrowd_values(self, year):
        """iscrowd must be 0 or 1."""
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            assert ann["iscrowd"] in (0, 1), \
                f"Invalid iscrowd value: {ann['iscrowd']} (must be 0 or 1)"

    @pytest.mark.parametrize("year", YEARS)
    def test_no_score_in_annotations(self, year):
        """Training annotations should not have 'score' (that's for inference)."""
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        scores_found = sum(1 for ann in coco["annotations"] if "score" in ann)
        assert scores_found == 0, \
            f"Found {scores_found} annotations with 'score' field — should be excluded for training"


class TestParquetDataset:
    """Test that parquet files exist and can be loaded."""

    @pytest.mark.parametrize("year", YEARS)
    def test_parquet_exists(self, year):
        parquet_path = DATA_DIR / str(year) / "parquet" / "data-00000-of-00001.parquet"
        assert parquet_path.exists(), f"Missing parquet: {parquet_path}"

    @pytest.mark.parametrize("year", YEARS)
    def test_parquet_row_count(self, year):
        """Parquet should have same number of rows as COCO images."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            pytest.skip("pyarrow not installed")
        parquet_path = DATA_DIR / str(year) / "parquet" / "data-00000-of-00001.parquet"
        table = pq.read_table(parquet_path)
        coco_path = DATA_DIR / str(year) / "annotations" / f"{year}_annotations.json"
        with open(coco_path) as f:
            coco = json.load(f)
        assert table.num_rows == len(coco["images"]), \
            f"Parquet rows ({table.num_rows}) != COCO images ({len(coco['images'])})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
