#!/usr/bin/env python3
"""Tile a canonical COCO dataset at 512x512 for the fixed-size test pipeline.

This is intentionally a separate step from downloading and format conversion.
It writes:

    <dataset>/tiles/<split>/images/*.jpg
    <dataset>/tiles/<split>/tile_manifest.json
    <dataset>/annotations/instances_<split>_tiles.json

The original full-image COCO is left untouched and remains the file to use for
evaluation after predictions are merged back to full-image coordinates.

Examples
--------
    python scripts/external/tile.py --dataset big_bird
    python scripts/external/tile.py --dataset new_mexico \
        --annotations annotations/instances_expert_refined.json --split expert
    python scripts/external/tile.py --dataset birddetector \
        --annotations annotations/instances_penguins_test.json --split penguins_test

Rules: a box is kept if its center is in the tile; it is clipped to the tile;
boxes with a clipped area below 30% of the original or a side below 4 px are
dropped.  Empty tiles are kept by default for evaluation.  Use
``--empty-policy cap`` when creating training tiles.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external.common import (
    dump_json,
    load_json,
    make_canonical_annotation,
    make_coco,
    resolve_path,
    utc_now,
    validate_coco,
)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def tile_origins(length: int, tile_size: int, stride: int) -> list[int]:
    """Origins that cover [0, length) with an aligned final tile."""
    if length <= 0:
        return []
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if length <= tile_size:
        return [0]
    origins = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def map_box_to_tile(
    bbox: Iterable[float],
    tile_x: int,
    tile_y: int,
    tile_width: int,
    tile_height: int,
    *,
    min_visibility: float = 0.3,
    min_side: float = 4.0,
) -> list[float] | None:
    """Map a full-image box into tile coordinates, applying the drop rules."""
    x, y, w, h = (float(v) for v in bbox)
    if w <= 0 or h <= 0:
        return None
    original_area = w * h
    x1 = max(x, float(tile_x))
    y1 = max(y, float(tile_y))
    x2 = min(x + w, float(tile_x + tile_width))
    y2 = min(y + h, float(tile_y + tile_height))
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    if clipped_w < min_side or clipped_h < min_side:
        return None
    if clipped_w * clipped_h < min_visibility * original_area:
        return None
    return [x1 - tile_x, y1 - tile_y, clipped_w, clipped_h]


def box_center_in_tile(
    bbox: Iterable[float], tile_x: int, tile_y: int, tile_width: int, tile_height: int
) -> bool:
    x, y, w, h = (float(v) for v in bbox)
    cx = x + w / 2.0
    cy = y + h / 2.0
    return tile_x <= cx < tile_x + tile_width and tile_y <= cy < tile_y + tile_height


# ---------------------------------------------------------------------------
# image readers
# ---------------------------------------------------------------------------


class ImageReader:
    width: int
    height: int

    def crop(self, x: int, y: int, width: int, height: int) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "ImageReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class PillowReader(ImageReader):
    def __init__(self, path: Path):
        from PIL import Image

        self._image = Image.open(path)
        self.width, self.height = self._image.size

    def crop(self, x: int, y: int, width: int, height: int):
        return self._image.crop((x, y, x + width, y + height))

    def close(self) -> None:
        self._image.close()


class RasterioReader(ImageReader):
    def __init__(self, path: Path):
        import rasterio
        from rasterio.windows import Window

        self._rasterio = rasterio
        self._Window = Window
        self._source = rasterio.open(path)
        self.width, self.height = self._source.width, self._source.height

    def crop(self, x: int, y: int, width: int, height: int):
        from PIL import Image
        import numpy as np

        window = self._Window(col_off=x, row_off=y, width=width, height=height)
        count = self._source.count
        if count >= 3:
            array = self._source.read([1, 2, 3], window=window)
            array = np.transpose(array, (1, 2, 0))
        else:
            array = self._source.read(1, window=window)
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.dtype != np.uint8:
            # GeoTIFFs in this suite are 8-bit RGB; normalise if they are not.
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="RGB")

    def close(self) -> None:
        self._source.close()


def open_reader(path: Path) -> ImageReader:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            return RasterioReader(path)
        except ImportError:
            # Small test TIFFs can still be handled by Pillow.  The West Africa
            # 2.2 GB mosaic should use rasterio so it is read window-by-window.
            return PillowReader(path)
    return PillowReader(path)


# ---------------------------------------------------------------------------
# tiling
# ---------------------------------------------------------------------------


def _tile_image_name(image_id: int, x0: int, y0: int, extension: str) -> str:
    return f"img{image_id:08d}_x{x0:05d}_y{y0:05d}.{extension}"


def _save_tile(image: Any, path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if args.format == "jpg":
        image.save(path, format="JPEG", quality=args.quality, optimize=True)
    else:
        image.save(path, format="PNG", optimize=True)


def _select_annotation_file(root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = sorted((root / "annotations").glob("instances_*.json"))
    all_file = root / "annotations" / "instances_all.json"
    if all_file.exists():
        return all_file
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no instances_*.json under {root / 'annotations'}")
    names = ", ".join(path.name for path in candidates)
    raise ValueError(f"multiple annotation files found; choose one with --annotations: {names}")


def _split_name(annotation_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    stem = annotation_path.stem
    if stem.startswith("instances_"):
        stem = stem[len("instances_") :]
    return stem or "all"


def _cap_empty_tiles(records: list[dict[str, Any]], ratio: float) -> set[int]:
    nonempty = [record for record in records if record["_num_annotations"] > 0]
    empty = [record for record in records if record["_num_annotations"] == 0]
    if not empty:
        return set()
    if nonempty:
        allowed = int(math.floor(len(nonempty) * ratio))
    else:
        allowed = 1
    if allowed >= len(empty):
        return {record["tile_id"] for record in empty}
    if allowed <= 0:
        return set()
    # Deterministic spread over the empty tiles rather than the first N.
    step = len(empty) / allowed
    keep = {empty[min(int(index * step), len(empty) - 1)]["tile_id"] for index in range(allowed)}
    return keep


def tile_dataset(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root) / args.dataset
    annotation_path = _select_annotation_file(root, args.annotations)
    split = _split_name(annotation_path, args.split)
    image_prefix = Path(args.img_prefix) if args.img_prefix else root
    out_dir = Path(args.out) if args.out else root / "tiles" / split
    tile_annotation_path = root / "annotations" / f"instances_{split}_tiles.json"
    manifest_path = out_dir / "tile_manifest.json"
    extension = args.format

    coco = load_json(annotation_path)
    source_images = coco.get("images", [])
    if args.limit is not None:
        source_images = source_images[: args.limit]
    annotations_by_image: dict[Any, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        annotations_by_image.setdefault(annotation.get("image_id"), []).append(annotation)

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    tiles: list[dict[str, Any]] = []
    next_annotation_id = 1
    dropped = 0

    for image_index, source_image in enumerate(source_images, 1):
        source_path = resolve_path(image_prefix, source_image["file_name"])
        if source_path is None:
            print(f"  skip missing image: {source_image['file_name']}", file=sys.stderr)
            continue
        with open_reader(source_path) as reader:
            width, height = reader.width, reader.height
            if width != int(source_image["width"]) or height != int(source_image["height"]):
                print(
                    f"  warning: {source_image['file_name']} metadata says "
                    f"{source_image['width']}x{source_image['height']}, file is {width}x{height}",
                    file=sys.stderr,
                )
            xs = tile_origins(width, args.tile_size, args.stride)
            ys = tile_origins(height, args.tile_size, args.stride)
            for x0 in xs:
                for y0 in ys:
                    tile_width = min(args.tile_size, width - x0)
                    tile_height = min(args.tile_size, height - y0)
                    tile_id = len(tiles) + 1
                    tile_name = _tile_image_name(int(source_image["id"]), x0, y0, extension)
                    relative_tile = f"images/{tile_name}"
                    tile_path = out_dir / relative_tile
                    _save_tile(reader.crop(x0, y0, tile_width, tile_height), tile_path, args)

                    tile_anns: list[dict[str, Any]] = []
                    for source_annotation in annotations_by_image.get(source_image["id"], []):
                        bbox = source_annotation.get("bbox")
                        if not bbox:
                            continue
                        if not box_center_in_tile(bbox, x0, y0, tile_width, tile_height):
                            continue
                        mapped = map_box_to_tile(
                            bbox,
                            x0,
                            y0,
                            tile_width,
                            tile_height,
                            min_visibility=args.min_visibility,
                            min_side=args.min_side,
                        )
                        if mapped is None:
                            dropped += 1
                            continue
                        tile_anns.append(
                            make_canonical_annotation(
                                next_annotation_id,
                                tile_id,
                                mapped,
                                original_category=source_annotation.get("original_category"),
                                original_id=source_annotation.get("id"),
                                extra={
                                    "source_annotation_id": source_annotation.get("id"),
                                    "point_derived": bool(source_annotation.get("point_derived", False)),
                                    "tile_synthetic_mask": True,
                                },
                            )
                        )
                        next_annotation_id += 1

                    tiles.append(
                        {
                            "tile_id": tile_id,
                            "source_image_id": source_image["id"],
                            "source_file_name": source_image["file_name"],
                            "x0": x0,
                            "y0": y0,
                            "width": tile_width,
                            "height": tile_height,
                            "file_name": relative_tile,
                            "num_annotations": len(tile_anns),
                            "_num_annotations": len(tile_anns),
                            "_annotations": tile_anns,
                        }
                    )
        if image_index % 10 == 0 or image_index == len(source_images):
            print(f"  {args.dataset}: {image_index}/{len(source_images)} source images")

    if args.empty_policy == "drop":
        keep_empty: set[int] = set()
    elif args.empty_policy == "cap":
        keep_empty = _cap_empty_tiles(tiles, args.max_empty_ratio)
    else:
        keep_empty = {record["tile_id"] for record in tiles if record["_num_annotations"] == 0}

    retained_tiles: list[dict[str, Any]] = []
    retained_annotations: list[dict[str, Any]] = []
    for record in tiles:
        if record["_num_annotations"] == 0 and record["tile_id"] not in keep_empty:
            tile_path = out_dir / record["file_name"]
            if tile_path.exists():
                tile_path.unlink()
            continue
        clean_record = {key: value for key, value in record.items() if not key.startswith("_")}
        retained_tiles.append(clean_record)
        if record["_num_annotations"] > 0:
            retained_annotations.extend(record["_annotations"])

    tile_images = [
        {
            "id": record["tile_id"],
            "file_name": record["file_name"],
            "width": record["width"],
            "height": record["height"],
            "source_image_id": record["source_image_id"],
            "source_file_name": record["source_file_name"],
            "tile_x": record["x0"],
            "tile_y": record["y0"],
        }
        for record in retained_tiles
    ]
    tile_coco = make_coco(
        tile_images,
        retained_annotations,
        info={
            "description": f"{args.dataset} {split} tiles",
            "source_annotation_file": str(annotation_path.relative_to(root)),
            "tile_size": args.tile_size,
            "stride": args.stride,
            "min_visibility": args.min_visibility,
            "min_side": args.min_side,
            "empty_policy": args.empty_policy,
            "generated_at": utc_now(),
        },
    )
    counts = validate_coco(tile_coco)
    dump_json(tile_annotation_path, tile_coco)
    manifest = {
        "dataset": args.dataset,
        "split": split,
        "source_annotation_file": str(annotation_path.relative_to(root)),
        "tile_annotation_file": str(tile_annotation_path.relative_to(root)),
        "tile_size": args.tile_size,
        "stride": args.stride,
        "min_visibility": args.min_visibility,
        "min_side": args.min_side,
        "empty_policy": args.empty_policy,
        "max_empty_ratio": args.max_empty_ratio,
        "counts": {
            "source_images": len(source_images),
            "tiles": counts["images"],
            "annotations": counts["annotations"],
            "empty_tiles": sum(1 for record in retained_tiles if record["num_annotations"] == 0),
            "dropped_boxes": dropped,
        },
        "tiles": retained_tiles,
        "generated_at": utc_now(),
    }
    dump_json(manifest_path, manifest)
    print(
        f"  wrote {tile_annotation_path} ({counts['images']} tiles, "
        f"{counts['annotations']} boxes)"
    )
    print(f"  wrote {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", default="data/external")
    parser.add_argument("--annotations", default=None, help="full-image COCO, relative to the dataset root")
    parser.add_argument("--split", default=None, help="output split name (default: derived from annotation file)")
    parser.add_argument("--img-prefix", default=None, help="root for image file names (default: dataset root)")
    parser.add_argument("--out", default=None, help="tile output directory (default: <dataset>/tiles/<split>)")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--min-visibility", type=float, default=0.3)
    parser.add_argument("--min-side", type=float, default=4.0)
    parser.add_argument("--empty-policy", choices=("keep", "drop", "cap"), default="keep")
    parser.add_argument("--max-empty-ratio", type=float, default=0.1)
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=None, help="tile only the first N source images")
    parser.add_argument("--clean", action="store_true", help="remove the tile output directory first")
    args = parser.parse_args(argv)

    if args.tile_size <= 0 or args.stride <= 0:
        parser.error("--tile-size and --stride must be positive")
    if args.stride > args.tile_size:
        parser.error("--stride must not exceed --tile-size")
    if not 0 <= args.min_visibility <= 1:
        parser.error("--min-visibility must be in [0, 1]")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if not 0 <= args.quality <= 100:
        parser.error("--quality must be in [0, 100]")
    if not 0 <= args.max_empty_ratio <= 1:
        parser.error("--max-empty-ratio must be in [0, 1]")
    try:
        tile_dataset(args)
    except Exception as error:  # noqa: BLE001
        print(f"tile.py: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
