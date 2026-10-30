"""Focused tests for external-dataset geometry and tiling rules."""

from __future__ import annotations

import unittest

from scripts.external.common import clip_bbox, polygon_bbox
from scripts.external.tile import box_center_in_tile, map_box_to_tile, tile_origins


class ClipBboxTests(unittest.TestCase):
    def test_clips_to_image(self):
        self.assertEqual(clip_bbox([-10, -5, 30, 20], 100, 100), [0, 0, 20, 15])

    def test_drops_degenerate(self):
        self.assertIsNone(clip_bbox([0, 0, 0, 10], 100, 100))
        self.assertIsNone(clip_bbox([200, 200, 10, 10], 100, 100))

    def test_min_visibility(self):
        # Half the box is outside the image.
        self.assertEqual(clip_bbox([-5, 0, 10, 10], 100, 100, min_visibility=0.3), [0, 0, 5, 10])
        self.assertIsNone(clip_bbox([-5, 0, 10, 10], 100, 100, min_visibility=0.6))

    def test_min_side(self):
        self.assertIsNone(clip_bbox([0, 0, 3, 20], 100, 100, min_side=4))


class PolygonBboxTests(unittest.TestCase):
    def test_flat_and_ring(self):
        self.assertEqual(polygon_bbox([10, 20, 30, 20, 30, 40]), [10, 20, 20, 20])
        self.assertEqual(polygon_bbox([[10, 20, 30, 20, 30, 40]]), [10, 20, 20, 20])

    def test_invalid(self):
        self.assertIsNone(polygon_bbox([]))
        self.assertIsNone(polygon_bbox([1, 2, 3]))


class TileGeometryTests(unittest.TestCase):
    def test_origins_cover_last_pixel(self):
        self.assertEqual(tile_origins(500, 512, 384), [0])
        self.assertEqual(tile_origins(1024, 512, 384), [0, 384, 512])
        self.assertEqual(tile_origins(1025, 512, 384)[-1], 513)

    def test_center_rule(self):
        self.assertTrue(box_center_in_tile([10, 10, 20, 20], 0, 0, 50, 50))
        self.assertFalse(box_center_in_tile([10, 10, 20, 20], 25, 0, 50, 50))

    def test_box_mapping_and_drop(self):
        mapped = map_box_to_tile([-10, -10, 40, 40], 0, 0, 100, 100, min_visibility=0.3)
        self.assertEqual(mapped, [0, 0, 30, 30])
        # Only 10% of the original area remains after clipping.
        self.assertIsNone(
            map_box_to_tile([-90, -90, 100, 100], 0, 0, 100, 100, min_visibility=0.3)
        )
        # A four-pixel side is the inclusive minimum.
        self.assertIsNotNone(map_box_to_tile([0, 0, 4, 20], 0, 0, 100, 100, min_side=4))
        self.assertIsNone(map_box_to_tile([0, 0, 3.9, 20], 0, 0, 100, 100, min_side=4))


if __name__ == "__main__":
    unittest.main()
