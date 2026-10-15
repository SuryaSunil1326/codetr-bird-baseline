#!/usr/bin/env python3
"""
upload_dataset.py

Uploads dataset folders to Hugging Face Hub using api.upload_folder.

Supports --resume to skip files that already exist on the Hub (by path),
and --dry-run to preview what would be uploaded without making changes.
"""

import argparse
from pathlib import Path
from huggingface_hub import CommitOperationDelete, HfApi


def clean_stray_root_files(api: HfApi, repo_id: str, repo_type: str, revision: str):
    """Delete stray files sitting at repo root while preserving Dataset/."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)
    except Exception:
        print("Repo empty or missing. Skipping cleanup.")
        return

    # Keep files inside Dataset/ and hidden .git paths
    files_to_delete = [
        f for f in files
        if not f.startswith(".git") and not f.startswith("Dataset/")
    ]

    if files_to_delete:
        operations = [CommitOperationDelete(path_in_repo=f) for f in files_to_delete]
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            operations=operations,
            commit_message="Cleanup: remove stray root-level files",
        )
        print(f"✓ Removed {len(files_to_delete)} stray root files.")
    else:
        print("No stray root files found.")


def get_existing_remote_files(api: HfApi, repo_id: str, repo_type: str, revision: str) -> set:
    """Return a set of file paths already present on the Hub branch."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)
        return set(files)
    except Exception:
        print("WARNING: Could not list remote files (repo may be empty). Assuming no existing files.")
        return set()


def filter_existing_files(local_dir: Path, existing_remote: set, path_in_repo: str) -> list:
    """Return only files that don't already exist on the Hub."""
    from pathlib import PurePosixPath
    base = PurePosixPath(path_in_repo) if path_in_repo else PurePosixPath("")
    all_files = sorted([f for f in local_dir.rglob("*") if f.is_file() and ".cache" not in f.parts])
    to_upload = []
    skipped = 0
    for file_path in all_files:
        rel_path = file_path.relative_to(local_dir)
        repo_path = (base / rel_path).as_posix()
        if repo_path in existing_remote:
            skipped += 1
        else:
            to_upload.append(file_path)
    if skipped:
        print(f"  Skipping {skipped} files already on Hub (--resume)")
    return to_upload


def main():
    p = argparse.ArgumentParser(description="Upload dataset folder to HF Hub")
    p.add_argument("--dataset-dir", required=True, type=Path, help="Local directory")
    p.add_argument("--repo-id", required=True, help="HF Repo ID")
    p.add_argument("--branch", default="main", help="Target branch")
    p.add_argument("--path-in-repo", default="Dataset", help="Path inside repo")
    p.add_argument("--clean-first", action="store_true", help="Clean stray root files first")
    p.add_argument("--resume", action="store_true", help="Skip files already on the Hub (match by path)")
    p.add_argument("--dry-run", action="store_true", help="Preview what would be uploading without making changes")

    args = p.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    if args.branch != "main":
        api.create_branch(repo_id=args.repo_id, repo_type="dataset", branch=args.branch, exist_ok=True)

    if args.clean_first:
        clean_stray_root_files(api, args.repo_id, "dataset", args.branch)

    # Count files
    all_files = sorted([f for f in args.dataset_dir.rglob("*") if f.is_file() and ".cache" not in f.parts])
    total_size_mb = sum(f.stat().st_size for f in all_files) / (1024 * 1024)
    print(f"Found {len(all_files)} files ({total_size_mb:.1f} MB total) in {args.dataset_dir}")

    if args.resume:
        existing_remote = get_existing_remote_files(api, args.repo_id, "dataset", args.branch)
        files_to_upload = filter_existing_files(args.dataset_dir, existing_remote, args.path_in_repo)
    else:
        files_to_upload = all_files

    if not files_to_upload:
        print("All files already exist on Hub. Nothing to upload.")
        return

    print(f"\nFiles to upload: {len(files_to_upload)}")
    upload_size_mb = sum(f.stat().st_size for f in files_to_upload) / (1024 * 1024)
    print(f"Upload size: {upload_size_mb:.1f} MB")

    if args.dry_run:
        print("\n[DRY RUN] Would upload the following files:")
        for f in files_to_upload[:20]:
            print(f"  {f.relative_to(args.dataset_dir)}")
        if len(files_to_upload) > 20:
            print(f"  ... and {len(files_to_upload) - 20} more")
        return

    print(f"\n🚀 Uploading {args.dataset_dir} -> {args.repo_id} (branch: {args.branch})...")

    api.upload_folder(
        folder_path=str(args.dataset_dir),
        path_in_repo=args.path_in_repo,
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.branch,
    )

    print("\n✓ Upload complete!")


if __name__ == "__main__":
    main()
