#!/usr/bin/env python3
"""Convert downloaded external datasets to the canonical single-class COCO.

The output is deliberately small and predictable::

    <root>/<dataset>/annotations/instances_*.json
    <root>/<dataset>/manifests/<annotation_stem>.json

Every image has a local ``file_name`` relative to ``<root>/<dataset>``.  Images
whose files were not downloaded are skipped, so a three-image sample stays a
three-image COCO file instead of referencing thousands of missing files.

West Africa needs rasterio because its single image is a GeoTIFF and its labels
are map coordinates, not pixels.  All other adapters use Pillow only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external.common import (
    category_names,
    clip_bbox,
    dataset_manifest,
    dump_json,
    image_size,
    is_empty_category,
    load_json,
    make_canonical_annotation,
    make_coco,
    polygon_bbox,
    resolve_path,
    utc_now,
    validate_coco,
)

DATASETS = (
    "big_bird",
    "izembek",
    "birddetector",
    "west_africa",
    "new_mexico",
    "mva2023",
)

BIG_BIRD_META = "raw/wilson-bigbird.json"
IZEMBEK_META = "raw/izembek-lagoon-birds-metadata.json"
WEST_AFRICA_CSV = "raw/labels_birds_full.csv"
WEST_AFRICA_TIF = "raw/seabirds_rgb.tif"
NEW_MEXICO_ROOT = "raw/uas-imagery-of-migratory-waterfowl"
NEW_MEXICO_JSON = {
    "expert": "experts/20230331_dronesforducks_expert_refined.json",
    "crowdsourced": "crowdsourced/20240220_dronesforducks_zooniverse_refined.json",
}


# ---------------------------------------------------------------------------
# generic COCO conversion helpers
# ---------------------------------------------------------------------------


def _source_category(categories: dict[Any, str], category_id: Any) -> str:
    return categories.get(category_id, str(category_id))


def _convert_annotations(
    source_annotations: Iterable[dict[str, Any]],
    id_map: dict[Any, int],
    images_by_id: dict[int, dict[str, Any]],
    categories: dict[Any, str],
    *,
    point_derived: bool = False,
    point_half_size: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    annotations: list[dict[str, Any]] = []
    dropped = {"empty": 0, "missing_image": 0, "invalid": 0}
    next_id = 1
    for source in source_annotations:
        source_image_id = source.get("image_id")
        image_id = id_map.get(source_image_id)
        if image_id is None:
            dropped["missing_image"] += 1
            continue
        category_name = _source_category(categories, source.get("category_id"))
        if is_empty_category(category_name):
            dropped["empty"] += 1
            continue
        image = images_by_id[image_id]
        bbox = source.get("bbox")
        if not bbox:
            bbox = polygon_bbox(source.get("segmentation"))
        if not bbox:
            dropped["invalid"] += 1
            continue
        clipped = clip_bbox(bbox, image["width"], image["height"])
        if clipped is None:
            dropped["invalid"] += 1
            continue
        segmentation = source.get("segmentation")
        if not isinstance(segmentation, list) or not segmentation:
            segmentation = None
        extra: dict[str, Any] = {}
        if point_derived:
            extra["point_derived"] = True
            if point_half_size is not None:
                extra["point_half_size"] = int(point_half_size)
        annotations.append(
            make_canonical_annotation(
                next_id,
                image_id,
                clipped,
                original_category=category_name,
                original_id=source.get("id"),
                segmentation=segmentation,
                iscrowd=int(source.get("iscrowd", 0) or 0),
                extra=extra,
            )
        )
        next_id += 1
    return annotations, dropped


def _write_result(
    root: Path,
    name: str,
    coco: dict[str, Any],
    *,
    output_name: str,
    source: str,
    license: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts = validate_coco(coco)
    annotation_path = root / "annotations" / output_name
    dump_json(annotation_path, coco)
    manifest = dataset_manifest(
        name=name,
        source=source,
        license=license,
        counts=counts,
        extra={
            "annotation_file": str(annotation_path.relative_to(root)),
            "generated_at": utc_now(),
            **(extra or {}),
        },
    )
    manifest_path = root / "manifests" / f"{Path(output_name).stem}.json"
    dump_json(manifest_path, manifest)
    print(f"  wrote {annotation_path} ({counts['images']} images, {counts['annotations']} boxes)")
    return manifest


# ---------------------------------------------------------------------------
# Big Bird
# ---------------------------------------------------------------------------


def prepare_big_bird(root: Path) -> dict[str, Any]:
    metadata = load_json(root / BIG_BIRD_META)
    categories = category_names(metadata.get("categories", []))
    source_images = metadata.get("images", [])
    images: list[dict[str, Any]] = []
    image_ids: dict[Any, int] = {}
    images_by_id: dict[int, dict[str, Any]] = {}

    for source in source_images:
        file_name = str(source.get("file_name", "")).replace("\\", "/")
        relative = file_name if file_name.startswith("images/") else f"images/{file_name}"
        candidate = resolve_path(root, relative)
        if candidate is None:
            continue
        try:
            width, height = image_size(candidate)
        except Exception:  # noqa: BLE001
            continue
        new_id = len(images) + 1
        image_ids[source.get("id")] = new_id
        record = {
            "id": new_id,
            "file_name": relative,
            "width": width,
            "height": height,
            "original_id": source.get("id"),
        }
        images.append(record)
        images_by_id[new_id] = record

    if not images:
        raise RuntimeError(f"no Big Bird images found under {root / 'images'}")

    annotations, dropped = _convert_annotations(
        metadata.get("annotations", []), image_ids, images_by_id, categories
    )
    coco = make_coco(
        images,
        annotations,
        info={
            "description": "Big Bird converted to single-class bird COCO",
            "source": "https://lila.science/datasets/big-bird/",
            "version": metadata.get("info", {}).get("version"),
            "license": metadata.get("info", {}).get("license"),
        },
        licenses=[metadata.get("info", {})] if metadata.get("info") else [],
    )
    return _write_result(
        root,
        "big_bird",
        coco,
        output_name="instances_all.json",
        source="https://lila.science/datasets/big-bird/",
        license="CC-BY 4.0",
        extra={"original_categories": categories, "dropped": dropped},
    )


# ---------------------------------------------------------------------------
# Izembek
# ---------------------------------------------------------------------------


def prepare_izembek(root: Path) -> dict[str, Any]:
    metadata = load_json(root / IZEMBEK_META)
    categories = category_names(metadata.get("categories", []))
    images: list[dict[str, Any]] = []
    image_ids: dict[Any, int] = {}
    images_by_id: dict[int, dict[str, Any]] = {}

    for source in metadata.get("images", []):
        file_name = str(source.get("file_name", "")).replace("\\", "/")
        if not file_name.startswith("images/"):
            file_name = "images/" + file_name
        candidate = resolve_path(root, file_name)
        if candidate is None:
            continue
        try:
            width, height = image_size(candidate)
        except Exception:  # noqa: BLE001
            continue
        new_id = len(images) + 1
        image_ids[source.get("id")] = new_id
        record = {
            "id": new_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "original_id": source.get("id"),
            "annotation_file": source.get("annotation_file"),
        }
        images.append(record)
        images_by_id[new_id] = record

    if not images:
        raise RuntimeError(f"no Izembek images found under {root / 'images'}")

    annotations, dropped = _convert_annotations(
        metadata.get("annotations", []), image_ids, images_by_id, categories
    )
    coco = make_coco(
        images,
        annotations,
        info={
            "description": "Izembek Lagoon waterfowl converted to single-class bird COCO",
            "source": "https://lila.science/datasets/izembek-lagoon-waterfowl/",
            "license": metadata.get("info", {}).get("license"),
        },
        licenses=[metadata.get("info", {})] if metadata.get("info") else [],
    )
    return _write_result(
        root,
        "izembek",
        coco,
        output_name="instances_all.json",
        source="https://lila.science/datasets/izembek-lagoon-waterfowl/",
        license="See LILA dataset page",
        extra={"original_categories": categories, "dropped": dropped},
    )


# ---------------------------------------------------------------------------
# BirdDetector
# ---------------------------------------------------------------------------


def _birddetector_split(
    root: Path,
    project_root: Path,
    csv_path: Path,
    split: str,
    project: str,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_image: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        image_path = str(row.get("image_path", "")).replace("\\", "/")
        if image_path:
            by_image.setdefault(image_path, []).append(row)

    images: list[dict[str, Any]] = []
    images_by_name: dict[str, dict[str, Any]] = {}
    for image_path in by_image:
        candidate = resolve_path(project_root, image_path)
        if candidate is None:
            continue
        try:
            width, height = image_size(candidate)
        except Exception:  # noqa: BLE001
            continue
        new_id = len(images) + 1
        relative = str(candidate.relative_to(root)).replace("\\", "/")
        record = {
            "id": new_id,
            "file_name": relative,
            "width": width,
            "height": height,
            "original_id": image_path,
        }
        images.append(record)
        images_by_name[image_path] = record

    if not images:
        return None, {"rows": len(rows), "images": 0, "annotations": 0, "missing_images": len(by_image)}

    annotations: list[dict[str, Any]] = []
    next_id = 1
    missing = 0
    for image_path, image_rows in by_image.items():
        image = images_by_name.get(image_path)
        if image is None:
            missing += len(image_rows)
            continue
        for row in image_rows:
            try:
                x1 = float(row.get("xmin", ""))
                y1 = float(row.get("ymin", ""))
                x2 = float(row.get("xmax", ""))
                y2 = float(row.get("ymax", ""))
            except (TypeError, ValueError):
                continue
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            clipped = clip_bbox([x, y, w, h], image["width"], image["height"])
            if clipped is None:
                continue
            annotations.append(
                make_canonical_annotation(
                    next_id,
                    image["id"],
                    clipped,
                    original_category=str(row.get("label", "Bird")),
                    original_id=None,
                )
            )
            next_id += 1

    coco = make_coco(
        images,
        annotations,
        info={
            "description": f"BirdDetector {project} {split} converted to single-class bird COCO",
            "source": "https://doi.org/10.5281/zenodo.5033174",
        },
        licenses=[],
    )
    counts = validate_coco(coco)
    return coco, {
        "rows": len(rows),
        "images": counts["images"],
        "annotations": counts["annotations"],
        "missing_images": missing,
    }


def prepare_birddetector(root: Path, project_filter: str | None = None) -> dict[str, Any]:
    raw = root / "raw"
    project_dirs = sorted({path.parent for path in raw.rglob("*_train.csv")})
    if project_filter:
        project_dirs = [path for path in project_dirs if path.name.casefold() == project_filter.casefold()]
    if not project_dirs:
        raise RuntimeError(f"no BirdDetector CSV files found under {raw}")

    results: dict[str, Any] = {}
    for project_root in project_dirs:
        project = project_root.name
        for csv_path in sorted(project_root.glob("*.csv")):
            split = "train" if csv_path.stem.casefold().endswith("_train") else "test"
            coco, counts = _birddetector_split(root, project_root, csv_path, split, project)
            if coco is None:
                print(f"  birddetector/{project}: no extracted images for {csv_path.name}; skipped")
                continue
            output_name = f"instances_{project}_{split}.json"
            manifest = _write_result(
                root,
                f"birddetector_{project}",
                coco,
                output_name=output_name,
                source="https://doi.org/10.5281/zenodo.5033174",
                license="See Zenodo record",
                extra={"project": project, "split": split, "csv": str(csv_path.relative_to(root)), **counts},
            )
            results[output_name] = manifest
    if not results:
        raise RuntimeError("no BirdDetector images were extracted; rerun download.py")
    return {"splits": results}


# ---------------------------------------------------------------------------
# West Africa
# ---------------------------------------------------------------------------


def west_africa_points_to_annotations(
    rows: Iterable[dict[str, str]],
    *,
    inverse_transform: Any,
    width: int,
    height: int,
    point_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pure-ish point conversion helper (inverse_transform is an Affine)."""
    annotations: list[dict[str, Any]] = []
    counts = {"rows": 0, "missing_coordinates": 0, "out_of_bounds": 0, "invalid": 0, "annotations": 0}
    half = point_size / 2.0
    next_id = 1
    for row in rows:
        counts["rows"] += 1
        try:
            x_map = float(row.get("X", ""))
            y_map = float(row.get("Y", ""))
        except (TypeError, ValueError):
            counts["missing_coordinates"] += 1
            continue
        col, row_px = inverse_transform * (x_map, y_map)
        x = float(col) - half
        y = float(row_px) - half
        clipped = clip_bbox([x, y, point_size, point_size], width, height)
        if clipped is None:
            # Distinguish points whose centre is outside the raster from bad boxes.
            if col < 0 or row_px < 0 or col >= width or row_px >= height:
                counts["out_of_bounds"] += 1
            else:
                counts["invalid"] += 1
            continue
        annotations.append(
            make_canonical_annotation(
                next_id,
                1,
                clipped,
                original_category=str(row.get("label", "(unknown)")),
                original_id=row.get("") or row.get("id"),
                extra={"point_derived": True, "point_half_size": int(half)},
            )
        )
        next_id += 1
        counts["annotations"] += 1
    return annotations, counts


def prepare_west_africa(root: Path, point_size: int = 32) -> dict[str, Any]:
    try:
        import rasterio
    except ImportError as error:
        raise RuntimeError(
            "West Africa needs rasterio to read the GeoTIFF geotransform. "
            "Install it in the external environment: pip install rasterio"
        ) from error

    csv_path = root / WEST_AFRICA_CSV
    tif_path = root / WEST_AFRICA_TIF
    if not tif_path.exists():
        raise RuntimeError(
            f"{tif_path} is missing; download it with "
            "scripts/external/download.py --dataset west_africa --full"
        )
    with rasterio.open(tif_path) as source:
        width, height = source.width, source.height
        inverse = ~source.transform
        transform = tuple(source.transform)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    annotations, counts = west_africa_points_to_annotations(
        rows,
        inverse_transform=inverse,
        width=width,
        height=height,
        point_size=point_size,
    )
    images = [
        {
            "id": 1,
            "file_name": WEST_AFRICA_TIF,
            "width": width,
            "height": height,
            "original_id": "seabirds_rgb.tif",
        }
    ]
    coco = make_coco(
        images,
        annotations,
        info={
            "description": "Aerial Seabirds West Africa point labels converted to synthetic boxes",
            "source": "https://lila.science/datasets/aerial-seabirds-west-africa/",
            "point_box_size": point_size,
            "transform": transform,
        },
        licenses=[{"name": "CDLA-Permissive-1.0"}],
    )
    return _write_result(
        root,
        "west_africa",
        coco,
        output_name="instances_all.json",
        source="https://lila.science/datasets/aerial-seabirds-west-africa/",
        license="CDLA-Permissive-1.0",
        extra={"point_box_size": point_size, "transform": transform, **counts},
    )


# ---------------------------------------------------------------------------
# New Mexico
# ---------------------------------------------------------------------------


def _find_new_mexico_json(root: Path, part: str) -> Path:
    direct = root / NEW_MEXICO_ROOT / NEW_MEXICO_JSON[part]
    if direct.exists():
        return direct
    matches = sorted((root / "raw").rglob(Path(NEW_MEXICO_JSON[part]).name))
    if not matches:
        raise RuntimeError(f"New Mexico {part} JSON not found under {root / 'raw'}")
    return matches[0]


def _prepare_new_mexico_part(root: Path, part: str) -> dict[str, Any]:
    json_path = _find_new_mexico_json(root, part)
    payload = load_json(json_path)
    categories = category_names(payload.get("categories", []))
    image_dir = json_path.parent / "images"
    images: list[dict[str, Any]] = []
    image_ids: dict[Any, int] = {}
    images_by_id: dict[int, dict[str, Any]] = {}

    for source in payload.get("images", []):
        file_name = str(source.get("file_name", "")).replace("\\", "/")
        candidate = resolve_path(image_dir, Path(file_name).name)
        if candidate is None:
            continue
        try:
            width, height = image_size(candidate)
        except Exception:  # noqa: BLE001
            continue
        new_id = len(images) + 1
        image_ids[source.get("id")] = new_id
        relative = str(candidate.relative_to(root)).replace("\\", "/")
        record = {
            "id": new_id,
            "file_name": relative,
            "width": width,
            "height": height,
            "original_id": source.get("id"),
        }
        images.append(record)
        images_by_id[new_id] = record

    if not images:
        raise RuntimeError(f"no New Mexico {part} images found under {image_dir}")

    annotations, dropped = _convert_annotations(
        payload.get("annotations", []), image_ids, images_by_id, categories
    )
    coco = make_coco(
        images,
        annotations,
        info={
            "description": f"New Mexico {part} converted to single-class bird COCO",
            "source": "https://lila.science/datasets/uas-imagery-of-migratory-waterfowl-at-new-mexico-wildlife-refuges/",
            "license": "CC-BY-NC 4.0",
        },
        licenses=[{"name": "CC-BY-NC 4.0"}],
    )
    output_name = f"instances_{part}_refined.json"
    manifest = _write_result(
        root,
        f"new_mexico_{part}",
        coco,
        output_name=output_name,
        source="https://lila.science/datasets/uas-imagery-of-migratory-waterfowl-at-new-mexico-wildlife-refuges/",
        license="CC-BY-NC 4.0",
        extra={"part": part, "original_categories": categories, "dropped": dropped},
    )
    return manifest


def prepare_new_mexico(root: Path, part: str = "expert") -> dict[str, Any]:
    parts = ("expert", "crowdsourced") if part == "both" else (part,)
    results = {name: _prepare_new_mexico_part(root, name) for name in parts}
    return {"parts": results}


# ---------------------------------------------------------------------------
# MVA2023
# ---------------------------------------------------------------------------


MVA_SPLITS = {
    "val_mini": "raw/data/drone2021/annotations/split_val_mini_coco.json",
    "train_mini": "raw/data/drone2021/annotations/split_train_mini_coco.json",
    "val": "raw/data/drone2021/annotations/split_val_coco.json",
    "train": "raw/data/drone2021/annotations/split_train_coco.json",
    "mva_val": "raw/data/mva2023_sod4bird_train/annotations/split_val_coco.json",
    "mva_train": "raw/data/mva2023_sod4bird_train/annotations/split_train_coco.json",
}


def _find_mva_json(root: Path, split: str) -> Path:
    direct = root / MVA_SPLITS[split]
    if direct.exists():
        return direct
    matches = sorted((root / "raw").rglob(direct.name))
    if not matches:
        raise RuntimeError(f"MVA2023 {split} annotation not found under {root / 'raw'}")
    return matches[0]


def prepare_mva2023(root: Path, split: str = "val_mini", allow_missing: bool = False) -> dict[str, Any]:
    json_path = _find_mva_json(root, split)
    payload = load_json(json_path)
    categories = category_names(payload.get("categories", []))
    # drone2021 images are extracted under raw/data/drone2021/images.  The
    # full MVA archive may use a different root; look in both places.
    image_roots = [
        root / "raw" / "data" / "drone2021" / "images",
        root / "raw" / "data" / "mva2023_sod4bird_train" / "images",
    ]
    images: list[dict[str, Any]] = []
    image_ids: dict[Any, int] = {}
    images_by_id: dict[int, dict[str, Any]] = {}
    for source in payload.get("images", []):
        file_name = str(source.get("file_name", "")).replace("\\", "/")
        candidate = None
        for image_root in image_roots:
            candidate = resolve_path(image_root, file_name)
            if candidate is not None:
                break
            candidate = resolve_path(image_root, Path(file_name).name)
            if candidate is not None:
                break
        if candidate is None:
            if not allow_missing:
                continue
            width = int(source.get("width", 0) or 0)
            height = int(source.get("height", 0) or 0)
            if width <= 0 or height <= 0:
                continue
            image_set = "mva2023_sod4bird_train" if split.startswith("mva_") else "drone2021"
            relative = str(Path("raw") / "data" / image_set / "images" / file_name).replace("\\", "/")
        else:
            try:
                width, height = image_size(candidate)
            except Exception:  # noqa: BLE001
                continue
            relative = str(candidate.relative_to(root)).replace("\\", "/")
        new_id = len(images) + 1
        image_ids[source.get("id")] = new_id
        record = {
            "id": new_id,
            "file_name": relative,
            "width": width,
            "height": height,
            "original_id": source.get("id"),
        }
        images.append(record)
        images_by_id[new_id] = record

    if not images:
        raise RuntimeError(
            f"no MVA2023 images found for {split}; download annotations only is not enough "
            "for tiling, use --mva-images or --full"
        )

    annotations, dropped = _convert_annotations(
        payload.get("annotations", []), image_ids, images_by_id, categories
    )
    coco = make_coco(
        images,
        annotations,
        info={
            "description": f"MVA2023 {split} converted to single-class bird COCO",
            "source": "https://github.com/takedarts/mva2023_sod4bird",
            "license": "See MVA2023 challenge terms",
        },
        licenses=[],
    )
    return _write_result(
        root,
        f"mva2023_{split}",
        coco,
        output_name=f"instances_{split}.json",
        source="https://github.com/takedarts/mva2023_sod4bird",
        license="See MVA2023 challenge terms",
        extra={"split": split, "original_categories": categories, "dropped": dropped},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_one(name: str, root: Path, args: argparse.Namespace) -> Any:
    if not root.exists():
        raise RuntimeError(f"{root} does not exist; run download.py first")
    if name == "big_bird":
        return prepare_big_bird(root)
    if name == "izembek":
        return prepare_izembek(root)
    if name == "birddetector":
        return prepare_birddetector(root, args.project)
    if name == "west_africa":
        return prepare_west_africa(root, args.point_box)
    if name == "new_mexico":
        return prepare_new_mexico(root, args.part)
    if name == "mva2023":
        return prepare_mva2023(root, args.split, allow_missing=args.allow_missing)
    raise AssertionError(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("all",) + DATASETS, default="all")
    parser.add_argument("--root", default="data/external")
    parser.add_argument("--project", default=None, help="BirdDetector project (default: all extracted projects)")
    parser.add_argument("--part", choices=("expert", "crowdsourced", "both"), default="expert")
    parser.add_argument("--split", choices=tuple(MVA_SPLITS), default="val_mini")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="MVA2023: write image records even if the image zip is not extracted yet",
    )
    parser.add_argument("--point-box", type=int, default=32, help="West Africa synthetic box size in pixels")
    args = parser.parse_args(argv)
    if args.point_box <= 0:
        parser.error("--point-box must be positive")

    names = DATASETS if args.dataset == "all" else (args.dataset,)
    failures: list[tuple[str, str]] = []
    for name in names:
        root = Path(args.root) / name
        if not root.exists():
            print(f"\n=== {name}: skipped (no {root}) ===")
            continue
        print(f"\n=== {name} ===")
        try:
            result = _run_one(name, root, args)
            if isinstance(result, dict) and "splits" in result:
                print(f"  {name}: {len(result['splits'])} split(s)")
            elif isinstance(result, dict) and "parts" in result:
                print(f"  {name}: {len(result['parts'])} part(s)")
        except Exception as error:  # noqa: BLE001
            failures.append((name, f"{type(error).__name__}: {error}"))
            print(f"  {name}: FAILED: {failures[-1][1]}", file=sys.stderr)
    if failures:
        print("\nFailures:", file=sys.stderr)
        for name, message in failures:
            print(f"  {name}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
