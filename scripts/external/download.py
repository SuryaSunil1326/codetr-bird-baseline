#!/usr/bin/env python3
"""Download small samples (or complete data) for the external bird datasets.

Examples
--------
    # tiny sample: metadata + three images
    python scripts/external/download.py --dataset big_bird --limit 3

    # download the one-file West Africa orthomosaic as well
    python scripts/external/download.py --dataset west_africa --full

    # only check URLs and sizes
    python scripts/external/download.py --dataset all --probe

The downloader and the converter are intentionally separate.  Downloads land
under ``data/external/<dataset>/`` and can be removed with ``--clean`` once the
conversion/tiling test has passed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external.common import (
    category_names,
    dataset_manifest,
    download_file,
    dump_json,
    image_size,
    is_empty_category,
    load_json,
    probe_url,
    sha256_file,
    url_join,
    utc_now,
)

DATASETS = (
    "big_bird",
    "izembek",
    "birddetector",
    "west_africa",
    "new_mexico",
    "mva2023",
)

BIG_BIRD_META_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/big-bird/wilson-bigbird.json"
)
BIG_BIRD_IMAGE_BASE = "https://storage.googleapis.com/public-datasets-lila/big-bird/"

IZEMBEK_META_URL = (
    "https://storage.googleapis.com/public-datasets-lila/izembek-lagoon-birds/"
    "izembek-lagoon-birds-metadata.zip"
)
IZEMBEK_IMAGE_BASE = "https://storage.googleapis.com/public-datasets-lila/izembek-lagoon-birds/"

ZENODO_RECORD = "https://zenodo.org/api/records/5033174"
ZENODO_FILE_BASE = "https://zenodo.org/api/records/5033174/files/"

WEST_AFRICA_CSV_URL = (
    "https://storage.googleapis.com/public-datasets-lila/aerial-birds-west-africa/"
    "labels_birds_full.csv"
)
WEST_AFRICA_TIF_URL = (
    "https://storage.googleapis.com/public-datasets-lila/aerial-birds-west-africa/seabirds_rgb.tif"
)

NEW_MEXICO_URL = (
    "https://storage.googleapis.com/public-datasets-lila/uas-imagery-of-migratory-waterfowl/"
    "uas-imagery-of-migratory-waterfowl.20240220.zip"
)

MVA2023_FOLDER = "https://drive.google.com/drive/folders/1vTHiIelagbzPO795yhOdNUFh9u2XxZP-"
MVA2023_MINI_ANNOTATIONS = (
    "data/drone2021/annotations/split_val_mini_coco.json",
    "data/drone2021/annotations/split_train_mini_coco.json",
)
MVA2023_SAMPLE_ANNOTATIONS = MVA2023_MINI_ANNOTATIONS + (
    "data/drone2021/annotations/split_val_coco.json",
    "data/drone2021/annotations/split_train_coco.json",
)
MVA2023_VAL_ANNOTATION = "data/drone2021/annotations/split_val_coco.json"
MVA2023_FULL_IMAGE_ZIP = "data/mva2023_sod4bird_train/images.zip"
MVA2023_FULL_ANNOTATIONS = (
    "data/mva2023_sod4bird_train/annotations/merged_train.json",
    "data/mva2023_sod4bird_train/annotations/split_train_coco.json",
    "data/mva2023_sod4bird_train/annotations/split_val_coco.json",
)


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------


def dataset_root(args: argparse.Namespace, name: str) -> Path:
    return Path(args.root) / name


def write_download_manifest(root: Path, manifest: dict[str, Any]) -> None:
    dump_json(root / "raw" / "download_manifest.json", manifest)


def _select_sample_images(
    images: Iterable[dict[str, Any]],
    annotations: Iterable[dict[str, Any]],
    categories: dict[Any, str],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Prefer images with birds, then fill with empty images."""
    images = list(images)
    annotations = list(annotations)
    by_image: dict[Any, list[dict[str, Any]]] = {}
    for annotation in annotations:
        by_image.setdefault(annotation.get("image_id"), []).append(annotation)

    def has_bird(image: dict[str, Any]) -> bool:
        for annotation in by_image.get(image.get("id"), []):
            name = categories.get(annotation.get("category_id"))
            if is_empty_category(name):
                continue
            if annotation.get("bbox") or annotation.get("segmentation"):
                return True
        return False

    nonempty = [image for image in images if has_bird(image)]
    empty = [image for image in images if not has_bird(image)]
    if limit is None:
        return nonempty + empty
    return (nonempty + empty)[: max(0, limit)]


def _safe_extract_members(
    zip_path: Path,
    dest_root: Path,
    members: Iterable[str],
) -> list[Path]:
    written: list[Path] = []
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for member in members:
            if member not in names:
                continue
            target = dest_root / member
            try:
                target.resolve().relative_to(dest_root.resolve())
            except ValueError:
                continue
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written.append(target)
    return written


def _extract_all(zip_path: Path, dest_root: Path) -> list[Path]:
    with zipfile.ZipFile(zip_path) as archive:
        return _safe_extract_members(zip_path, dest_root, archive.namelist())


def _member_by_name(archive: zipfile.ZipFile, wanted: str) -> str | None:
    wanted = wanted.replace("\\", "/")
    names = archive.namelist()
    if wanted in names:
        return wanted
    wanted_base = Path(wanted).name.casefold()
    matches = [name for name in names if Path(name).name.casefold() == wanted_base]
    if not matches:
        return None
    # Prefer a member whose full path ends with the requested relative path.
    suffix_matches = [
        name for name in matches if name.replace("\\", "/").casefold().endswith(wanted.casefold())
    ]
    return (suffix_matches or matches)[0]


def _read_csv_member(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with archive.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        return list(csv.DictReader(text))


# ---------------------------------------------------------------------------
# Big Bird
# ---------------------------------------------------------------------------


def download_big_bird(args: argparse.Namespace) -> dict[str, Any]:
    root = dataset_root(args, "big_bird")
    raw = root / "raw"
    images_dir = root / "images"
    raw.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = download_file(BIG_BIRD_META_URL, raw / "wilson-bigbird.json", overwrite=args.overwrite)
    metadata = load_json(metadata_path)
    selected = _select_sample_images(
        metadata.get("images", []),
        metadata.get("annotations", []),
        category_names(metadata.get("categories", [])),
        None if args.full else args.limit,
    )
    files: list[dict[str, Any]] = []
    for index, image in enumerate(selected, 1):
        file_name = str(image["file_name"]).replace("\\", "/")
        relative = f"images/{file_name}"
        path = root / relative
        if not path.exists():
            download_file(url_join(BIG_BIRD_IMAGE_BASE, file_name), path, overwrite=args.overwrite)
        width, height = image_size(path)
        files.append(
            {
                "file_name": relative,
                "source_url": url_join(BIG_BIRD_IMAGE_BASE, file_name),
                "width": width,
                "height": height,
                "sha256": sha256_file(path),
            }
        )
        if index % 25 == 0:
            print(f"  big_bird: {index}/{len(selected)} images")
    manifest = dataset_manifest(
        name="big_bird",
        source="https://lila.science/datasets/big-bird/",
        license="CC-BY 4.0 (see LILA dataset page)",
        counts={"images": len(files), "source_images": len(selected)},
        extra={
            "downloaded_at": utc_now(),
            "metadata_url": BIG_BIRD_META_URL,
            "image_base": BIG_BIRD_IMAGE_BASE,
            "full": bool(args.full),
            "files": files,
        },
    )
    write_download_manifest(root, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Izembek
# ---------------------------------------------------------------------------


def download_izembek(args: argparse.Namespace) -> dict[str, Any]:
    root = dataset_root(args, "izembek")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    metadata_zip = download_file(IZEMBEK_META_URL, raw / "izembek-lagoon-birds-metadata.zip", overwrite=args.overwrite)
    metadata_path = raw / "izembek-lagoon-birds-metadata.json"
    if not metadata_path.exists() or args.overwrite:
        with zipfile.ZipFile(metadata_zip) as archive:
            json_members = [name for name in archive.namelist() if name.endswith(".json")]
            if not json_members:
                raise RuntimeError("Izembek metadata zip contains no JSON file")
            with archive.open(json_members[0]) as src, open(metadata_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    metadata = load_json(metadata_path)
    selected = _select_sample_images(
        metadata.get("images", []),
        metadata.get("annotations", []),
        category_names(metadata.get("categories", [])),
        None if args.full else args.limit,
    )
    files: list[dict[str, Any]] = []
    for index, image in enumerate(selected, 1):
        file_name = str(image["file_name"]).replace("\\", "/")
        if not file_name.startswith("images/"):
            file_name = "images/" + file_name
        path = root / file_name
        if not path.exists():
            download_file(url_join(IZEMBEK_IMAGE_BASE, file_name), path, overwrite=args.overwrite)
        width, height = image_size(path)
        files.append(
            {
                "file_name": file_name,
                "source_url": url_join(IZEMBEK_IMAGE_BASE, file_name),
                "width": width,
                "height": height,
                "sha256": sha256_file(path),
            }
        )
        if index % 25 == 0:
            print(f"  izembek: {index}/{len(selected)} images")
    manifest = dataset_manifest(
        name="izembek",
        source="https://lila.science/datasets/izembek-lagoon-waterfowl/",
        license="See LILA dataset page",
        counts={"images": len(files), "source_images": len(selected)},
        extra={
            "downloaded_at": utc_now(),
            "metadata_url": IZEMBEK_META_URL,
            "image_base": IZEMBEK_IMAGE_BASE,
            "full": bool(args.full),
            "files": files,
        },
    )
    write_download_manifest(root, manifest)
    return manifest


# ---------------------------------------------------------------------------
# BirdDetector (Zenodo)
# ---------------------------------------------------------------------------


def download_birddetector(args: argparse.Namespace) -> dict[str, Any]:
    root = dataset_root(args, "birddetector")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    project = args.project
    zip_path = raw / f"{project}.zip"

    if not zip_path.exists() or args.overwrite:
        record = json.load(urllib.request.urlopen(ZENODO_RECORD, timeout=60))
        files = {entry["key"]: entry for entry in record.get("files", [])}
        key = f"{project}.zip"
        if key not in files:
            available = ", ".join(sorted(files))
            raise SystemExit(f"Zenodo project {project!r} not found. Available: {available}")
        download_file(files[key]["links"]["self"], zip_path, overwrite=args.overwrite)

    project_root = raw / project
    project_root.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        csv_members = [
            name
            for name in names
            if Path(name).suffix.lower() == ".csv"
            and Path(name).name.casefold().startswith(project.casefold())
        ]
        if not csv_members:
            csv_members = [name for name in names if Path(name).suffix.lower() == ".csv"]

        if args.full:
            extracted_paths = _extract_all(zip_path, project_root)
            for path in extracted_paths:
                if path.is_file():
                    extracted.append({"file_name": str(path.relative_to(root)), "sha256": sha256_file(path)})
        else:
            # Keep both official splits represented in a small download.
            members_to_extract: list[str] = list(csv_members)
            for csv_member in csv_members:
                rows = _read_csv_member(archive, csv_member)
                image_paths: list[str] = []
                seen: set[str] = set()
                for row in rows:
                    image_path = str(row.get("image_path", "")).replace("\\", "/")
                    if image_path and image_path not in seen:
                        seen.add(image_path)
                        image_paths.append(image_path)
                for image_path in image_paths[: args.limit]:
                    member = _member_by_name(archive, image_path)
                    if member:
                        members_to_extract.append(member)
            members_to_extract = list(dict.fromkeys(members_to_extract))
            for path in _safe_extract_members(zip_path, project_root, members_to_extract):
                if path.is_file():
                    extracted.append({"file_name": str(path.relative_to(root)), "sha256": sha256_file(path)})

    manifest = dataset_manifest(
        name=f"birddetector_{project}",
        source="https://doi.org/10.5281/zenodo.5033174",
        license="See Zenodo record",
        counts={"files": len(extracted)},
        extra={
            "downloaded_at": utc_now(),
            "project": project,
            "archive": str(zip_path.relative_to(root)),
            "archive_sha256": sha256_file(zip_path),
            "full": bool(args.full),
            "files": extracted,
        },
    )
    write_download_manifest(root, manifest)
    return manifest


# ---------------------------------------------------------------------------
# West Africa
# ---------------------------------------------------------------------------


def download_west_africa(args: argparse.Namespace) -> dict[str, Any]:
    root = dataset_root(args, "west_africa")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    csv_path = download_file(WEST_AFRICA_CSV_URL, raw / "labels_birds_full.csv", overwrite=args.overwrite)
    files.append({"file_name": str(csv_path.relative_to(root)), "sha256": sha256_file(csv_path)})

    if args.full or args.west_africa_image:
        tif_path = download_file(WEST_AFRICA_TIF_URL, raw / "seabirds_rgb.tif", overwrite=args.overwrite)
        files.append({"file_name": str(tif_path.relative_to(root)), "sha256": sha256_file(tif_path)})
    else:
        print(
            "  west_africa: downloaded the point CSV only; the 2.2 GB GeoTIFF is "
            "downloaded with --full (or --west-africa-image)."
        )

    manifest = dataset_manifest(
        name="west_africa",
        source="https://lila.science/datasets/aerial-seabirds-west-africa/",
        license="CDLA-Permissive-1.0",
        counts={"files": len(files), "image_downloaded": bool(args.full or args.west_africa_image)},
        extra={
            "downloaded_at": utc_now(),
            "csv_url": WEST_AFRICA_CSV_URL,
            "tif_url": WEST_AFRICA_TIF_URL,
            "full": bool(args.full),
            "files": files,
        },
    )
    write_download_manifest(root, manifest)
    return manifest


# ---------------------------------------------------------------------------
# New Mexico
# ---------------------------------------------------------------------------


def _new_mexico_parts(part: str) -> list[tuple[str, str]]:
    choices = {
        "expert": ("experts", "20230331_dronesforducks_expert_refined.json"),
        "crowdsourced": ("crowdsourced", "20240220_dronesforducks_zooniverse_refined.json"),
    }
    if part == "both":
        return [choices["expert"], choices["crowdsourced"]]
    return [choices[part]]


def download_new_mexico(args: argparse.Namespace) -> dict[str, Any]:
    root = dataset_root(args, "new_mexico")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = download_file(NEW_MEXICO_URL, raw / "newmexico.zip", overwrite=args.overwrite)
    extracted: list[dict[str, Any]] = []

    if args.full:
        paths = _extract_all(zip_path, raw)
        extracted = [
            {"file_name": str(path.relative_to(root)), "sha256": sha256_file(path)}
            for path in paths
            if path.is_file()
        ]
    else:
        with zipfile.ZipFile(zip_path) as archive:
            members_to_extract: list[str] = []
            for folder, json_name in _new_mexico_parts(args.part):
                json_member = _member_by_name(archive, f"{folder}/{json_name}")
                if not json_member:
                    # The archive may use a top-level dataset directory.
                    json_member = next(
                        (
                            name
                            for name in archive.namelist()
                            if name.replace("\\", "/").endswith(f"{folder}/{json_name}")
                        ),
                        None,
                    )
                if not json_member:
                    raise RuntimeError(f"New Mexico archive lacks {folder}/{json_name}")
                members_to_extract.append(json_member)
                payload = json.loads(archive.read(json_member))
                selected = _select_sample_images(
                    payload.get("images", []),
                    payload.get("annotations", []),
                    category_names(payload.get("categories", [])),
                    args.limit,
                )
                for image in selected:
                    image_name = Path(str(image["file_name"])).name
                    image_member = _member_by_name(archive, f"{folder}/images/{image_name}")
                    if not image_member:
                        # Fall back to a basename search anywhere under the part.
                        image_member = next(
                            (
                                name
                                for name in archive.namelist()
                                if name.replace("\\", "/").endswith(f"/{folder}/images/{image_name}")
                            ),
                            None,
                        )
                    if image_member:
                        members_to_extract.append(image_member)
            members_to_extract = list(dict.fromkeys(members_to_extract))
            paths = _safe_extract_members(zip_path, raw, members_to_extract)
            extracted = [
                {"file_name": str(path.relative_to(root)), "sha256": sha256_file(path)}
                for path in paths
                if path.is_file()
            ]

    manifest = dataset_manifest(
        name="new_mexico",
        source="https://lila.science/datasets/uas-imagery-of-migratory-waterfowl-at-new-mexico-wildlife-refuges/",
        license="CC-BY-NC 4.0",
        counts={"files": len(extracted)},
        extra={
            "downloaded_at": utc_now(),
            "archive": str(zip_path.relative_to(root)),
            "archive_sha256": sha256_file(zip_path),
            "part": args.part,
            "full": bool(args.full),
            "files": extracted,
        },
    )
    write_download_manifest(root, manifest)
    return manifest


# ---------------------------------------------------------------------------
# MVA2023 (Google Drive)
# ---------------------------------------------------------------------------


def _gdown_files() -> dict[str, dict[str, Any]]:
    try:
        import gdown
    except ImportError as error:
        raise SystemExit(
            "MVA2023 needs gdown. Install it in your external environment with:\n"
            "  pip install gdown"
        ) from error
    entries = gdown.download_folder(
        MVA2023_FOLDER,
        skip_download=True,
        quiet=True,
        use_cookies=False,
    )
    return {str(entry.path): entry for entry in entries}


def _gdown_one(entry: Any, destination: Path, overwrite: bool) -> Path:
    import gdown

    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET gdown:{entry.path}\n      -> {destination}")
    gdown.download(id=entry.id, output=str(destination), quiet=False)
    return destination


def download_mva2023(args: argparse.Namespace) -> dict[str, Any]:
    root = dataset_root(args, "mva2023")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    entries = _gdown_files()

    wanted_paths = list(MVA2023_SAMPLE_ANNOTATIONS)
    if args.full:
        wanted_paths = list(MVA2023_FULL_ANNOTATIONS) + [MVA2023_FULL_IMAGE_ZIP]
    elif args.mva_images:
        # The mini annotation files name their image folders (e.g. 6/..., 9/...).
        wanted_paths = list(MVA2023_SAMPLE_ANNOTATIONS)

    files: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for remote_path in wanted_paths:
        entry = entries.get(remote_path)
        if not entry:
            print(f"  mva2023: {remote_path} is not present in the Drive folder; skipped")
            continue
        destination = raw / remote_path
        try:
            _gdown_one(entry, destination, args.overwrite)
            files.append({"file_name": str(destination.relative_to(root)), "sha256": sha256_file(destination)})
        except Exception as error:  # noqa: BLE001 - quota errors should not lose annotations
            message = f"{type(error).__name__}: {error}"
            failed.append({"path": remote_path, "error": message})
            print(f"  mva2023: could not download {remote_path}: {message}", file=sys.stderr)

    # Download the image zips matching the mini annotation prefixes.  This is
    # opt-in because an individual zip can still be hundreds of megabytes.
    if args.mva_images and not args.full:
        mini_paths = [raw / path for path in MVA2023_MINI_ANNOTATIONS]
        prefixes: set[str] = set()
        for path in mini_paths:
            if not path.exists():
                continue
            payload = load_json(path)
            for image in payload.get("images", []):
                file_name = str(image.get("file_name", "")).replace("\\", "/")
                if "/" in file_name:
                    prefixes.add(file_name.split("/", 1)[0])
        for prefix in sorted(prefixes):
            remote_path = f"data/drone2021/images/{prefix}.zip"
            entry = entries.get(remote_path)
            if not entry:
                print(f"  mva2023: image zip {remote_path} not found; skipped")
                continue
            destination = raw / remote_path
            try:
                _gdown_one(entry, destination, args.overwrite)
                files.append({"file_name": str(destination.relative_to(root)), "sha256": sha256_file(destination)})
                _extract_all(destination, raw / "data" / "drone2021" / "images")
            except Exception as error:  # noqa: BLE001 - Google Drive quota is common
                message = f"{type(error).__name__}: {error}"
                failed.append({"path": remote_path, "error": message})
                print(f"  mva2023: could not download {remote_path}: {message}", file=sys.stderr)

    if not args.mva_images and not args.full:
        print(
            "  mva2023: annotations only. Use --mva-images to fetch the mini "
            "image zips, or --full for the full training archive."
        )

    if args.full and (raw / MVA2023_FULL_IMAGE_ZIP).exists():
        _extract_all(raw / MVA2023_FULL_IMAGE_ZIP, raw / "data" / "mva2023_sod4bird_train" / "images")

    manifest = dataset_manifest(
        name="mva2023",
        source=MVA2023_FOLDER,
        license="See MVA2023 challenge terms",
        counts={"files": len(files), "failed": len(failed)},
        extra={
            "downloaded_at": utc_now(),
            "folder": MVA2023_FOLDER,
            "full": bool(args.full),
            "mva_images": bool(args.mva_images),
            "files": files,
            "failed": failed,
        },
    )
    write_download_manifest(root, manifest)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def probe_all() -> None:
    urls = {
        "big_bird_metadata": BIG_BIRD_META_URL,
        "big_bird_image_example": BIG_BIRD_IMAGE_BASE + "1195.jpg",
        "izembek_metadata": IZEMBEK_META_URL,
        "izembek_image_example": IZEMBEK_IMAGE_BASE
        + "images/2017_Replicate_2017-09-30_Cam1_293A0014.JPG",
        "birddetector_penguins": ZENODO_FILE_BASE + "penguins.zip/content",
        "west_africa_csv": WEST_AFRICA_CSV_URL,
        "west_africa_tif": WEST_AFRICA_TIF_URL,
        "new_mexico_zip": NEW_MEXICO_URL,
        "mva2023_folder": MVA2023_FOLDER,
    }
    for name, url in urls.items():
        info = probe_url(url)
        print(name, json.dumps(info, sort_keys=True))


def clean_datasets(args: argparse.Namespace) -> None:
    names = DATASETS if args.dataset == "all" else (args.dataset,)
    for name in names:
        path = dataset_root(args, name)
        if path.exists():
            print(f"removing {path}")
            shutil.rmtree(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("all",) + DATASETS, default="all")
    parser.add_argument("--root", default="data/external", help="cache root (default: data/external)")
    parser.add_argument("--limit", type=int, default=3, help="images per split/project in sample mode (default: 3)")
    parser.add_argument("--full", action="store_true", help="download the full dataset instead of a sample")
    parser.add_argument("--overwrite", action="store_true", help="replace existing files")
    parser.add_argument("--probe", action="store_true", help="print URL/status/size information and exit")
    parser.add_argument("--clean", action="store_true", help="remove this dataset's local cache and exit")
    parser.add_argument("--project", default="penguins", help="BirdDetector project zip (default: penguins)")
    parser.add_argument("--part", choices=("expert", "crowdsourced", "both"), default="expert")
    parser.add_argument(
        "--west-africa-image",
        action="store_true",
        help="download the 2.2 GB West Africa GeoTIFF without downloading everything else",
    )
    parser.add_argument(
        "--mva-images",
        action="store_true",
        help="for MVA2023, also download the mini image zips (may be large)",
    )
    args = parser.parse_args(argv)

    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.probe:
        probe_all()
        return 0
    if args.clean:
        clean_datasets(args)
        return 0

    names = DATASETS if args.dataset == "all" else (args.dataset,)
    failures: list[tuple[str, str]] = []
    for name in names:
        print(f"\n=== {name} ===")
        try:
            if name == "big_bird":
                manifest = download_big_bird(args)
            elif name == "izembek":
                manifest = download_izembek(args)
            elif name == "birddetector":
                manifest = download_birddetector(args)
            elif name == "west_africa":
                manifest = download_west_africa(args)
            elif name == "new_mexico":
                manifest = download_new_mexico(args)
            elif name == "mva2023":
                manifest = download_mva2023(args)
            else:  # pragma: no cover - argparse constrains this
                raise AssertionError(name)
            print(f"  {name}: {manifest['counts']}")
        except Exception as error:  # noqa: BLE001 - report every dataset and continue
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
