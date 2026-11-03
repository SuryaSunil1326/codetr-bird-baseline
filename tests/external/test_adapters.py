"""Small end-to-end tests for the external conversion adapters and tiler."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.external.common import dump_json
from scripts.external.prepare import (
    prepare_big_bird,
    prepare_birddetector,
    prepare_izembek,
    prepare_mva2023,
    prepare_new_mexico,
    west_africa_points_to_annotations,
)
from scripts.external import tile


def make_image(path: Path, size=(100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(30, 120, 200)).save(path)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class AdapterTests(unittest.TestCase):
    def test_big_bird_drops_empty_category_and_uses_polygon_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "big_bird"
            make_image(root / "images" / "1.jpg")
            make_image(root / "images" / "2.jpg")
            dump_json(
                root / "raw" / "wilson-bigbird.json",
                {
                    "info": {"version": "test"},
                    "images": [
                        {"id": "1.jpg", "file_name": "1.jpg", "width": 100, "height": 100},
                        {"id": "2.jpg", "file_name": "2.jpg", "width": 100, "height": 100},
                    ],
                    "annotations": [
                        {"id": "a", "image_id": "1.jpg", "category_id": 0},
                        {
                            "id": "b",
                            "image_id": "1.jpg",
                            "category_id": 1,
                            "bbox": [10, 20, 5, 6],
                        },
                        {
                            "id": "c",
                            "image_id": "2.jpg",
                            "category_id": 1,
                            "segmentation": [[50, 50, 60, 50, 60, 60, 50, 60]],
                        },
                    ],
                    "categories": [
                        {"id": 0, "name": "empty"},
                        {"id": 1, "name": "great egret"},
                    ],
                },
            )
            prepare_big_bird(root)
            coco = read_json(root / "annotations" / "instances_all.json")
            self.assertEqual(len(coco["images"]), 2)
            self.assertEqual(len(coco["annotations"]), 2)
            self.assertEqual(coco["annotations"][0]["original_category"], "great egret")
            self.assertEqual(coco["annotations"][0]["bbox"], [10, 20, 5, 6])
            self.assertEqual(coco["annotations"][1]["bbox"], [50, 50, 10, 10])
            self.assertTrue(coco["annotations"][1]["segmentation"])

    def test_izembek_drops_empty_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "izembek"
            make_image(root / "images" / "frame.JPG")
            dump_json(
                root / "raw" / "izembek-lagoon-birds-metadata.json",
                {
                    "images": [
                        {"id": "frame.JPG", "file_name": "images/frame.JPG", "width": 100, "height": 100}
                    ],
                    "annotations": [
                        {"id": "a", "image_id": "frame.JPG", "category_id": 0, "bbox": [1, 2, 3, 4]},
                        {"id": "b", "image_id": "frame.JPG", "category_id": 5},
                    ],
                    "categories": [{"id": 0, "name": "Brant"}, {"id": 5, "name": "Empty"}],
                },
            )
            prepare_izembek(root)
            coco = read_json(root / "annotations" / "instances_all.json")
            self.assertEqual(len(coco["annotations"]), 1)
            self.assertEqual(coco["images"][0]["file_name"], "images/frame.JPG")

    def test_birddetector_csv_to_xywh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "birddetector"
            project = root / "raw" / "penguins"
            make_image(project / "foo.png")
            (project / "penguins_train.csv").write_text(
                "image_path,xmin,ymin,xmax,ymax,label\nfoo.png,10,20,30,50,Bird\n",
                encoding="utf-8",
            )
            prepare_birddetector(root)
            coco = read_json(root / "annotations" / "instances_penguins_train.json")
            self.assertEqual(coco["annotations"][0]["bbox"], [10, 20, 20, 30])

    def test_new_mexico_case_insensitive_image_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "new_mexico"
            part = root / "raw" / "uas-imagery-of-migratory-waterfowl" / "experts"
            make_image(part / "images" / "frame.jpg")
            dump_json(
                part / "20230331_dronesforducks_expert_refined.json",
                {
                    "images": [{"id": 1, "file_name": "frame.JPG", "width": 100, "height": 100}],
                    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]}],
                    "categories": [{"id": 1, "name": "Mallard"}],
                },
            )
            prepare_new_mexico(root, "expert")
            coco = read_json(root / "annotations" / "instances_expert_refined.json")
            self.assertEqual(len(coco["images"]), 1)
            self.assertEqual(coco["annotations"][0]["original_category"], "Mallard")

    def test_mva_category_zero_is_remapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mva2023"
            make_image(root / "raw" / "data" / "drone2021" / "images" / "6" / "363.jpg")
            dump_json(
                root / "raw" / "data" / "drone2021" / "annotations" / "split_val_mini_coco.json",
                {
                    "images": [{"id": 363, "file_name": "6/363.jpg", "width": 100, "height": 100}],
                    "annotations": [
                        {"id": 0, "image_id": 363, "category_id": 0, "bbox": [1, 2, 3, 4]}
                    ],
                    "categories": [{"id": 0, "name": "bird"}],
                },
            )
            prepare_mva2023(root, "val_mini")
            coco = read_json(root / "annotations" / "instances_val_mini.json")
            self.assertEqual(coco["annotations"][0]["category_id"], 1)
            self.assertEqual(coco["annotations"][0]["bbox"], [1, 2, 3, 4])

    def test_west_africa_point_to_box(self):
        class Inverse:
            def __mul__(self, xy):
                return (xy[0] / 0.01, (305.691 - xy[1]) / 0.01)

        rows = [
            {"": "1", "label": "Royal tern", "X": "5.0", "Y": "100.0"},
            {"": "2", "label": "Royal tern", "X": "NA", "Y": "100.0"},
            {"": "3", "label": "Royal tern", "X": "9999.0", "Y": "100.0"},
        ]
        annotations, counts = west_africa_points_to_annotations(
            rows, inverse_transform=Inverse(), width=1000, height=30000, point_size=32
        )
        self.assertEqual(len(annotations), 1)
        expected = [484.0, 20553.1, 32.0, 32.0]
        for actual, wanted in zip(annotations[0]["bbox"], expected):
            self.assertAlmostEqual(actual, wanted, places=4)
        self.assertTrue(annotations[0]["point_derived"])
        self.assertEqual(counts["missing_coordinates"], 1)
        self.assertEqual(counts["out_of_bounds"], 1)


class TileTests(unittest.TestCase):
    def test_tiling_writes_manifest_and_tile_coco(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demo"
            make_image(root / "images" / "frame.jpg", size=(100, 80))
            dump_json(
                root / "annotations" / "instances_all.json",
                {
                    "images": [{"id": 1, "file_name": "images/frame.jpg", "width": 100, "height": 80}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
                        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [90, 70, 10, 10]},
                    ],
                    "categories": [{"id": 1, "name": "bird", "supercategory": "bird"}],
                },
            )
            code = tile.main(
                [
                    "--dataset",
                    "demo",
                    "--root",
                    tmp,
                    "--tile-size",
                    "64",
                    "--stride",
                    "32",
                    "--empty-policy",
                    "keep",
                ]
            )
            self.assertEqual(code, 0)
            coco = read_json(root / "annotations" / "instances_all_tiles.json")
            manifest = read_json(root / "tiles" / "all" / "tile_manifest.json")
            self.assertGreater(len(coco["images"]), 1)
            self.assertEqual(len(coco["images"]), len(manifest["tiles"]))
            self.assertGreater(len(coco["annotations"]), 0)
            for tile_record in manifest["tiles"]:
                self.assertTrue((root / "tiles" / "all" / tile_record["file_name"]).exists())
            self.assertTrue((root / "tiles" / "all" / "images").is_dir())

    def test_empty_cap_drops_some_empty_tiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demo"
            make_image(root / "images" / "frame.jpg", size=(100, 80))
            dump_json(
                root / "annotations" / "instances_all.json",
                {
                    "images": [{"id": 1, "file_name": "images/frame.jpg", "width": 100, "height": 80}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}
                    ],
                    "categories": [{"id": 1, "name": "bird", "supercategory": "bird"}],
                },
            )
            code = tile.main(
                [
                    "--dataset",
                    "demo",
                    "--root",
                    tmp,
                    "--tile-size",
                    "64",
                    "--stride",
                    "32",
                    "--empty-policy",
                    "cap",
                    "--max-empty-ratio",
                    "0",
                ]
            )
            self.assertEqual(code, 0)
            manifest = read_json(root / "tiles" / "all" / "tile_manifest.json")
            self.assertTrue(all(record["num_annotations"] > 0 for record in manifest["tiles"]))


if __name__ == "__main__":
    unittest.main()
