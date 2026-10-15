#!/usr/bin/env python3
"""Comprehensive unit tests for COCO conversion scripts."""

import pytest
import json
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from json_to_coco import convert_json_to_coco



# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_coco_input():
    """Create a valid COCO input structure for testing."""
    return {
        "images": [
            {
                "id": 1,
                "file_name": "test_image_1.jpg",
                "width": 640,
                "height": 480
            },
            {
                "id": 2,
                "file_name": "test_image_2.jpg",
                "width": 800,
                "height": 600
            }
        ],
        "annotations": [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 300],  # x, y, width, height
                "area": 60000,
                "iscrowd": 0,
                "score": 0.95,
                "segmentation": [[10, 10, 50, 50, 60, 60, 70, 70]]  # quad polygon
            },
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [300, 200, 150, 100],
                "area": 15000,
                "iscrowd": 0,
                "score": 0.87,
                "segmentation": [[300, 200, 450, 200, 450, 300, 300, 300]]
            },
            {
                "image_id": 2,
                "category_id": 1,
                "bbox": [50, 50, 100, 100],
                "area": 10000,
                "iscrowd": 0,
                "score": 0.92,
                "segmentation": [[50, 50, 150, 50, 150, 150, 50, 150]]
            }
        ],
        "categories": [
            {"id": 1, "name": "Bird", "supercategory": "none"}
        ]
    }


@pytest.fixture
def temp_coco_file(sample_coco_input):
    """Create a temporary COCO JSON file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_coco_input, f)
        temp_path = f.name
    yield Path(temp_path)
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# Tests for convert_json_to_coco function
# ============================================================================

class TestConvertJsonToCoco:
    """Tests for the convert_json_to_coco function."""

    def test_preserves_segmentation_field(self, temp_coco_file):
        """Test that segmentation field is preserved in output annotations."""
        output_path = Path(str(temp_coco_file) + "_out.json")

        try:
            convert_json_to_coco(temp_coco_file, output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            # Verify all annotations have segmentation field
            assert "annotations" in result
            assert len(result["annotations"]) == 3

            for ann in result["annotations"]:
                assert "segmentation" in ann, f"Missing segmentation in annotation {ann.get('id')}"
                assert isinstance(ann["segmentation"], list), "Segmentation should be a list"
                assert len(ann["segmentation"]) > 0, "Segmentation should not be empty"

                # Each segmentation should be a list of polygons
                for polygon in ann["segmentation"]:
                    assert isinstance(polygon, list), "Each polygon should be a list"
                    assert len(polygon) >= 8, "Polygon should have at least 8 coordinates (4 points)"
                    assert len(polygon) % 2 == 0, "Polygon should have even number of coordinates"

        finally:
            if output_path.exists():
                os.unlink(output_path)

    def test_excludes_score_by_default(self, temp_coco_file):
        """Test that score field is excluded when include_score=False (default)."""
        output_path = Path(str(temp_coco_file) + "_out.json")

        try:
            convert_json_to_coco(temp_coco_file, output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            for ann in result["annotations"]:
                assert "score" not in ann, f"Score should not be present when include_score=False, but found in {ann}"

        finally:
            if output_path.exists():
                os.unlink(output_path)

    def test_includes_score_when_flag_set(self, temp_coco_file):
        """Test that score field is included when include_score=True."""
        output_path = Path(str(temp_coco_file) + "_out.json")

        try:
            convert_json_to_coco(temp_coco_file, output_path, include_score=True)

            with open(output_path) as f:
                result = json.load(f)

            for ann in result["annotations"]:
                assert "score" in ann, f"Score should be present when include_score=True, but missing in {ann}"
                assert isinstance(ann["score"], (int, float)), "Score should be numeric"
                assert 0 <= ann["score"] <= 1, "Score should be between 0 and 1"

        finally:
            if output_path.exists():
                os.unlink(output_path)

    def test_preserves_bbox_format(self, temp_coco_file):
        """Test that bbox is in correct COCO format [x, y, width, height]."""
        output_path = Path(str(temp_coco_file) + "_out.json")

        try:
            convert_json_to_coco(temp_coco_file, output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            for ann in result["annotations"]:
                assert "bbox" in ann, "Missing bbox field"
                bbox = ann["bbox"]
                assert isinstance(bbox, list), "Bbox should be a list"
                assert len(bbox) == 4, "Bbox should have 4 elements [x, y, width, height]"
                assert all(isinstance(v, (int, float)) for v in bbox), "All bbox values should be numeric"
                assert bbox[2] > 0, "Width should be positive"
                assert bbox[3] > 0, "Height should be positive"

        finally:
            if output_path.exists():
                os.unlink(output_path)

    def test_preserves_area_field(self, temp_coco_file):
        """Test that area field is preserved and correct."""
        output_path = Path(str(temp_coco_file) + "_out.json")

        try:
            convert_json_to_coco(temp_coco_file, output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            for ann in result["annotations"]:
                assert "area" in ann, "Missing area field"
                assert isinstance(ann["area"], (int, float)), "Area should be numeric"
                assert ann["area"] > 0, "Area should be positive"

                # Verify area matches bbox width * height
                bbox = ann["bbox"]
                expected_area = bbox[2] * bbox[3]
                assert abs(ann["area"] - expected_area) < 0.01, f"Area {ann['area']} doesn't match bbox {expected_area}"

        finally:
            if output_path.exists():
                os.unlink(output_path)

    def test_preserves_iscrowd_field(self, temp_coco_file):
        """Test that iscrowd field is preserved."""
        output_path = Path(str(temp_coco_file) + "_out.json")

        try:
            convert_json_to_coco(temp_coco_file, output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            for ann in result["annotations"]:
                assert "iscrowd" in ann, "Missing iscrowd field"
                assert ann["iscrowd"] in (0, 1), "iscrowd should be 0 or 1"

        finally:
            if output_path.exists():
                os.unlink(output_path)

    def test_preserves_category_id(self, temp_coco_file):
        """Test that category_id is preserved."""
        output_path = Path(str(temp_coco_file) + "_out.json")

        try:
            convert_json_to_coco(temp_coco_file, output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            for ann in result["annotations"]:
                assert "category_id" in ann, "Missing category_id field"
                assert ann["category_id"] == 1, "Category ID should be 1 (Bird)"

        finally:
            if output_path.exists():
                os.unlink(output_path)

    def test_preserves_image_structure(self, temp_coco_file):
        """Test that images array is preserved correctly."""
        output_path = Path(str(temp_coco_file) + "_out.json")

        try:
            convert_json_to_coco(temp_coco_file, output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            assert "images" in result
            assert len(result["images"]) == 2

            for img in result["images"]:
                assert "id" in img
                assert "file_name" in img
                assert "width" in img
                assert "height" in img
                assert isinstance(img["width"], int)
                assert isinstance(img["height"], int)
                assert img["width"] > 0
                assert img["height"] > 0

        finally:
            if output_path.exists():
                os.unlink(output_path)

    def test_preserves_categories(self, temp_coco_file):
        """Test that categories are preserved."""
        output_path = Path(str(temp_coco_file) + "_out.json")

        try:
            convert_json_to_coco(temp_coco_file, output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            assert "categories" in result
            assert len(result["categories"]) == 1
            cat = result["categories"][0]
            assert cat["id"] == 1
            assert cat["name"] == "Bird"
            assert cat["supercategory"] == "none"

        finally:
            if output_path.exists():
                os.unlink(output_path)

    def test_annotation_ids_are_unique(self, temp_coco_file):
        """Test that annotation IDs are unique and sequential."""
        output_path = Path(str(temp_coco_file) + "_out.json")

        try:
            convert_json_to_coco(temp_coco_file, output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            ann_ids = [ann["id"] for ann in result["annotations"]]
            assert len(ann_ids) == len(set(ann_ids)), "Annotation IDs should be unique"
            assert ann_ids == list(range(1, len(ann_ids) + 1)), "Annotation IDs should be sequential starting from 1"

        finally:
            if output_path.exists():
                os.unlink(output_path)

    def test_image_ids_are_unique(self, temp_coco_file):
        """Test that image IDs are unique."""
        output_path = Path(str(temp_coco_file) + "_out.json")

        try:
            convert_json_to_coco(temp_coco_file, output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            img_ids = [img["id"] for img in result["images"]]
            assert len(img_ids) == len(set(img_ids)), "Image IDs should be unique"

        finally:
            if output_path.exists():
                os.unlink(output_path)

    def test_output_is_valid_json(self, temp_coco_file):
        """Test that output is valid JSON and can be loaded."""
        output_path = Path(str(temp_coco_file) + "_out.json")

        try:
            convert_json_to_coco(temp_coco_file, output_path, include_score=False)

            # Should not raise JSONDecodeError
            with open(output_path) as f:
                result = json.load(f)

            # Verify all required top-level keys exist
            required_keys = ["images", "annotations", "categories"]
            for key in required_keys:
                assert key in result, f"Missing required top-level key: {key}"

        finally:
            if output_path.exists():
                os.unlink(output_path)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_annotations(self):
        """Test handling of images with no annotations."""
        empty_coco = {
            "images": [{"id": 1, "file_name": "empty.jpg", "width": 640, "height": 480}],
            "annotations": [],
            "categories": [{"id": 1, "name": "Bird", "supercategory": "none"}]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(empty_coco, f)
            temp_path = f.name

        output_path = Path(temp_path + "_out.json")

        try:
            convert_json_to_coco(Path(temp_path), output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            assert len(result["annotations"]) == 0
            assert len(result["images"]) == 1

        finally:
            for p in [temp_path, output_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_missing_segmentation_in_input(self):
        """Test handling when input annotations lack segmentation field."""
        coco_no_seg = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 200, 300],
                    "area": 60000,
                    "iscrowd": 0,
                    "score": 0.95
                    # No segmentation field
                }
            ],
            "categories": [{"id": 1, "name": "Bird", "supercategory": "none"}]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_no_seg, f)
            temp_path = f.name

        output_path = Path(temp_path + "_out.json")

        try:
            convert_json_to_coco(Path(temp_path), output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            # Should have empty segmentation list when not provided
            assert "segmentation" in result["annotations"][0]
            assert result["annotations"][0]["segmentation"] == []

        finally:
            for p in [temp_path, output_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_multiple_polygons_in_segmentation(self):
        """Test handling of multiple polygons in segmentation."""
        coco_multi_poly = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 200, 300],
                    "area": 60000,
                    "iscrowd": 0,
                    "score": 0.95,
                    "segmentation": [
                        [10, 10, 50, 50, 60, 60, 70, 70],  # polygon 1
                        [80, 80, 120, 80, 120, 120, 80, 120]  # polygon 2
                    ]
                }
            ],
            "categories": [{"id": 1, "name": "Bird", "supercategory": "none"}]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_multi_poly, f)
            temp_path = f.name

        output_path = Path(temp_path + "_out.json")

        try:
            convert_json_to_coco(Path(temp_path), output_path, include_score=False)

            with open(output_path) as f:
                result = json.load(f)

            seg = result["annotations"][0]["segmentation"]
            assert isinstance(seg, list)
            assert len(seg) == 2  # Two polygons
            for poly in seg:
                assert isinstance(poly, list)
                assert len(poly) >= 8

        finally:
            for p in [temp_path, output_path]:
                if os.path.exists(p):
                    os.unlink(p)


# ============================================================================
# Integration Test (if build_coco.py is available)
# ============================================================================

class TestBuildCocoIntegration:
    """Integration tests that verify build_coco.py output can be processed."""

    def test_build_coco_output_is_valid_input(self):
        """Test that build_coco.py output can be used as input to json_to_coco.py."""
        # This test would require the actual dataset directory
        # Skipping for now - can be enabled when dataset is available
        pytest.skip("Requires actual dataset directory")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
