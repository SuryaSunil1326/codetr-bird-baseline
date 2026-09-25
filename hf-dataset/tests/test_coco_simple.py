#!/usr/bin/env python3
"""Simple test to verify COCO conversion works correctly."""
import json
import tempfile
import os
from pathlib import Path

# Add parent dir to path so we can import json_to_coco
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from json_to_coco import convert_json_to_coco


def test_chester_to_coco():
    """Test converting Chester Island JSON to COCO format."""
    
    # Create a sample Chester Island JSON input
    chester_input = {
        "images": [
            {
                "file_name": "test_image_1.jpg",
                "width": 640,
                "height": 480
            },
            {
                "file_name": "test_image_2.jpg",
                "width": 800,
                "height": 600
            }
        ]
    }
    
    # Add detections to the first image
    chester_input["images"][0]["detections"] = [
        {
            "bbox": [100, 100, 200, 300],  # x_min, y_min, x_max, y_max
            "score": 0.95,
            "category_id": 1,
            "category_name": "Bird",
            "tcws_species": "ROTEA",
            "segmentation": [[10, 10, 50, 50, 60, 60, 70, 70]]  # quad polygon
        }
    ]
    
    # Add detections to the second image
    chester_input["images"][1]["detections"] = [
        {
            "bbox": [50, 50, 150, 150],
            "score": 0.87,
            "category_id": 1,
            "category_name": "Bird",
            "tcws_species": "ROTEA",
            "segmentation": [[300, 200, 450, 200, 450, 300, 300, 300]]
        }
    ]
    
    # Create temp input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(chester_input, f)
        input_path = f.name
    
    # Create temp output file
    output_path = Path(input_path + "_out.json")
    
    try:
        # Test 1: Convert without score (default)
        convert_json_to_coco(Path(input_path), output_path, include_score=False)
        
        with open(output_path) as f:
            result = json.load(f)
        
        print("Test 1: Convert without score (default)")
        print(f"  Images: {len(result['images'])}")
        print(f"  Annotations: {len(result['annotations'])}")
        print(f"  Categories: {len(result['categories'])}")
        
        # Verify annotations have segmentation
        for ann in result["annotations"]:
            assert "segmentation" in ann, "Missing segmentation in annotation"
            assert isinstance(ann["segmentation"], list), "Segmentation should be a list"
            assert len(ann["segmentation"]) > 0, "Segmentation should not be empty"
            print(f"  Annotation {ann['id']}: segmentation preserved ✓")
        
        # Verify score is excluded
        for ann in result["annotations"]:
            assert "score" not in ann, f"Score should not be present when include_score=False"
        print("  Score excluded by default ✓")
        
        # Test 2: Convert with score flag
        convert_json_to_coco(Path(input_path), output_path, include_score=True)
        
        with open(output_path) as f:
            result2 = json.load(f)
        
        print("\nTest 2: Convert with include_score=True")
        print(f"  Images: {len(result2['images'])}")
        print(f"  Annotations: {len(result2['annotations'])}")
        
        # Verify score is included
        for ann in result2["annotations"]:
            assert "score" in ann, f"Score should be present when include_score=True"
            print(f"  Annotation {ann['id']}: score={ann['score']} ✓")
        
        print("\n✓ All tests passed!")
        
    finally:
        # Cleanup
        if os.path.exists(input_path):
            os.unlink(input_path)
        if output_path.exists():
            os.unlink(output_path)


if __name__ == "__main__":
    test_chester_to_coco()