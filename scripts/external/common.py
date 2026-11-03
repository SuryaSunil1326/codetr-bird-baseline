"""Shared helpers for the external bird-dataset scripts.

These scripts deliberately depend on the standard library plus Pillow.  Only the
West-Africa GeoTIFF path needs rasterio, and that import is kept lazy so the
other datasets work in a minimal environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

CANONICAL_CATEGORY = {"id": 1, "name": "bird", "supercategory": "bird"}
USER_AGENT = "codetr-bird-baseline-external/1.0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# files and JSON
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(text: str, max_length: int = 80) -> str:
    """Make a filesystem-safe stem while keeping it recognisable."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return (text[:max_length] or "item").strip("._-")


def resolve_path(root: str | os.PathLike[str], relative: str | os.PathLike[str]) -> Path | None:
    """Resolve a path under *root*, matching case-insensitively if needed.

    Dataset archives occasionally disagree on ``.JPG`` vs ``.jpg``.  This keeps
    the converter usable on case-sensitive Linux filesystems.
    """
    root = Path(root)
    relative = Path(str(relative).replace("\\", "/"))
    if relative.is_absolute():
        return relative if relative.exists() else None
    current = root
    for part in relative.parts:
        if part in ("", "."):
            continue
        candidate = current / part
        if candidate.exists():
            current = candidate
            continue
        try:
            matches = [p for p in current.iterdir() if p.name.casefold() == part.casefold()]
        except FileNotFoundError:
            return None
        if not matches:
            return None
        current = matches[0]
    return current if current.exists() else None


def copy_or_link(src: Path, dst: Path, link: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _request(url: str, method: str = "GET", headers: dict[str, str] | None = None):
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    return urllib.request.Request(url, method=method, headers=request_headers)


def probe_url(url: str, timeout: int = 30) -> dict[str, Any]:
    """Return basic HTTP metadata without downloading a large response."""
    try:
        with urllib.request.urlopen(_request(url, method="HEAD"), timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return {
                "url": url,
                "status": getattr(response, "status", None),
                "content_length": int(length) if length and length.isdigit() else None,
                "content_type": response.headers.get("Content-Type"),
                "final_url": response.geturl(),
            }
    except Exception as head_error:  # noqa: BLE001 - probe must not be fatal
        try:
            with urllib.request.urlopen(
                _request(url, headers={"Range": "bytes=0-0"}), timeout=timeout
            ) as response:
                content_range = response.headers.get("Content-Range", "")
                total = None
                if "/" in content_range:
                    tail = content_range.rsplit("/", 1)[-1]
                    total = int(tail) if tail.isdigit() else None
                return {
                    "url": url,
                    "status": getattr(response, "status", None),
                    "content_length": total,
                    "content_type": response.headers.get("Content-Type"),
                    "final_url": response.geturl(),
                    "head_error": str(head_error),
                }
        except Exception as range_error:  # noqa: BLE001
            return {"url": url, "error": f"{type(range_error).__name__}: {range_error}"}


def download_file(
    url: str,
    dest: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    timeout: int = 120,
    quiet: bool = False,
) -> Path:
    """Download *url* to *dest* atomically, skipping an existing file."""
    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    if not quiet:
        print(f"  GET {url}\n      -> {dest}")
    with urllib.request.urlopen(_request(url), timeout=timeout) as response, open(tmp, "wb") as f:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        written = 0
        last_report = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
            if not quiet and total and written - last_report >= max(total // 10, 1):
                print(f"      {written / (1 << 20):.1f}/{total / (1 << 20):.1f} MiB")
                last_report = written
    if dest.exists() and overwrite:
        dest.unlink()
    tmp.replace(dest)
    if not quiet:
        print(f"      done ({dest.stat().st_size / (1 << 20):.1f} MiB)")
    return dest


def url_join(base: str, relative: str | os.PathLike[str]) -> str:
    relative = str(relative).replace("\\", "/").lstrip("/")
    return base.rstrip("/") + "/" + urllib.parse.quote(relative, safe="/")


# ---------------------------------------------------------------------------
# image and COCO helpers
# ---------------------------------------------------------------------------


def image_size(path: str | os.PathLike[str]) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return int(image.width), int(image.height)


def clip_bbox(
    bbox: Iterable[float],
    width: float,
    height: float,
    *,
    min_side: float = 0.0,
    min_visibility: float = 0.0,
) -> list[float] | None:
    """Clip ``[x, y, w, h]`` to an image and enforce size/visibility rules."""
    x, y, w, h = (float(v) for v in bbox)
    if w <= 0 or h <= 0:
        return None
    original_area = w * h
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + w)
    y2 = min(float(height), y + h)
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    if clipped_w < min_side or clipped_h < min_side:
        return None
    if min_visibility > 0 and (clipped_w * clipped_h) < min_visibility * original_area:
        return None
    return [x1, y1, clipped_w, clipped_h]


def polygon_bbox(polygon: Any) -> list[float] | None:
    """Return a bbox for a COCO polygon (flat or list-of-rings)."""
    if not polygon:
        return None
    rings = polygon
    if rings and isinstance(rings[0], (int, float)):
        rings = [rings]
    xs: list[float] = []
    ys: list[float] = []
    for ring in rings:
        if not isinstance(ring, (list, tuple)) or len(ring) < 4:
            continue
        values = [float(v) for v in ring]
        xs.extend(values[0::2])
        ys.extend(values[1::2])
    if not xs or not ys:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def rect_polygon(bbox: Iterable[float]) -> list[list[float]]:
    x, y, w, h = (float(v) for v in bbox)
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def is_empty_category(name: str | None, category_id: Any = None) -> bool:
    if name is not None and str(name).strip().casefold() in {"empty", "no bird", "none"}:
        return True
    return False


def category_names(categories: Iterable[dict[str, Any]]) -> dict[Any, str]:
    return {category.get("id"): str(category.get("name", "")) for category in categories}


def make_canonical_annotation(
    annotation_id: int,
    image_id: int,
    bbox: Iterable[float],
    *,
    original_category: str | None = None,
    original_id: Any = None,
    segmentation: Any = None,
    iscrowd: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bbox = [float(v) for v in bbox]
    annotation = {
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": CANONICAL_CATEGORY["id"],
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": int(iscrowd),
        "segmentation": segmentation if segmentation else rect_polygon(bbox),
    }
    if not segmentation:
        annotation["synthetic_mask"] = True
    if original_category is not None:
        annotation["original_category"] = original_category
    if original_id is not None:
        annotation["original_id"] = original_id
    if extra:
        annotation.update(extra)
    return annotation


def make_coco(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    info: dict[str, Any] | None = None,
    licenses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "info": info or {},
        "licenses": licenses or [],
        "images": images,
        "annotations": annotations,
        "categories": [dict(CANONICAL_CATEGORY)],
    }


def validate_coco(coco: dict[str, Any], *, require_images: bool = True) -> dict[str, int]:
    """Validate the parts of COCO that the MMDetection dataset relies on."""
    images = coco.get("images")
    annotations = coco.get("annotations")
    categories = coco.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO must contain 'images' and 'annotations' lists")
    if categories != [CANONICAL_CATEGORY]:
        raise ValueError(f"categories must be exactly [{CANONICAL_CATEGORY}], got {categories!r}")
    if require_images and not images:
        raise ValueError("COCO has no images")

    image_ids: set[int] = set()
    for image in images:
        image_id = image.get("id")
        if not isinstance(image_id, int):
            raise ValueError(f"image id is not an integer: {image_id!r}")
        if image_id in image_ids:
            raise ValueError(f"duplicate image id: {image_id}")
        image_ids.add(image_id)
        for key in ("file_name", "width", "height"):
            if key not in image:
                raise ValueError(f"image {image_id} lacks {key!r}")
        if int(image["width"]) <= 0 or int(image["height"]) <= 0:
            raise ValueError(f"image {image_id} has invalid dimensions")

    annotation_ids: set[int] = set()
    for annotation in annotations:
        annotation_id = annotation.get("id")
        if not isinstance(annotation_id, int):
            raise ValueError(f"annotation id is not an integer: {annotation_id!r}")
        if annotation_id in annotation_ids:
            raise ValueError(f"duplicate annotation id: {annotation_id}")
        annotation_ids.add(annotation_id)
        if annotation.get("image_id") not in image_ids:
            raise ValueError(f"annotation {annotation_id} references unknown image")
        if annotation.get("category_id") != CANONICAL_CATEGORY["id"]:
            raise ValueError(f"annotation {annotation_id} has an unexpected category")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"annotation {annotation_id} has invalid bbox")
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"annotation {annotation_id} has a degenerate bbox")
        annotation.setdefault("area", float(bbox[2]) * float(bbox[3]))
        annotation.setdefault("iscrowd", 0)
    return {"images": len(images), "annotations": len(annotations)}


def dataset_manifest(
    *,
    name: str,
    source: str,
    license: str | None,
    counts: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "dataset": name,
        "source": source,
        "license": license,
        "generated_at": utc_now(),
        "counts": counts,
    }
    if extra:
        manifest.update(extra)
    return manifest
