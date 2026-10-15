"""
upload_images_hf_batched.py

Uploads large image directories to Hugging Face Hub using multi-file batch commits.
Chunks by total payload size (~150 MB) to balance socket stability and HF commit limits.
"""

import argparse
import sys
from pathlib import Path, PurePosixPath
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

LARGE_FILE_THRESHOLD = 50 * 1024 * 1024  # 50 MB single-file standalone commit
TARGET_BATCH_BYTES = 150 * 1024 * 1024   # ~150 MB per batch payload


def clean_repo(api: HfApi, repo_id: str, repo_type: str, revision: str):
    """Delete all existing files in a single atomic commit."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)
    except Exception:
        print("Repo is empty or doesn't exist yet. Skipping cleanup.")
        return

    files_to_delete = [f for f in files if not f.startswith(".git")]

    if files_to_delete:
        operations = [CommitOperationDelete(path_in_repo=f) for f in files_to_delete]
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            operations=operations,
            commit_message="Cleanup: wipe branch before fresh upload",
        )
        print(f"✓ Deleted {len(files_to_delete)} files in 1 commit.")
    else:
        print("Branch is already clean.")


def create_batches_by_size(all_files: list, target_bytes: int = TARGET_BATCH_BYTES):
    """Group files into batches where total size is <= target_bytes."""
    batches = []
    current_batch = []
    current_size = 0

    for file_path in all_files:
        file_size = file_path.stat().st_size

        # If single file exceeds large threshold, it gets handled separately
        if file_size > LARGE_FILE_THRESHOLD:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            batches.append([file_path])
            continue

        if current_size + file_size > target_bytes and current_batch:
            batches.append(current_batch)
            current_batch = [file_path]
            current_size = file_size
        else:
            current_batch.append(file_path)
            current_size += file_size

    if current_batch:
        batches.append(current_batch)

    return batches


def upload_batch(
    repo_id: str,
    local_dir: Path,
    repo_type: str,
    revision: str,
    path_in_repo: str,
    target_mb: int = 150,
    clean_first: bool = False,
):
    """Upload images in size-capped batches to avoid socket timeouts & HF commit limits."""
    api = HfApi()

    if clean_first:
        clean_repo(api, repo_id, repo_type, revision)

    base = PurePosixPath(path_in_repo) if path_in_repo else PurePosixPath("")

    all_files = sorted([f for f in local_dir.rglob("*") if f.is_file() and ".cache" not in f.parts])

    if not all_files:
        print("No files found.")
        return

    target_bytes = target_mb * 1024 * 1024
    batches = create_batches_by_size(all_files, target_bytes=target_bytes)
    total_size_mb = sum(f.stat().st_size for f in all_files) / (1024 * 1024)
    print(f"Found {len(all_files)} files ({total_size_mb:.1f} MB total).")
    print(f"Grouped into {len(batches)} target-size batches (~{target_mb} MB/batch)...")

    for idx, chunk in enumerate(batches, 1):
        operations = []
        large_files = []
        batch_bytes = sum(f.stat().st_size for f in chunk)

        for file_path in chunk:
            rel_path = file_path.relative_to(local_dir)
            repo_file_path = (base / rel_path).as_posix()

            if file_path.stat().st_size > LARGE_FILE_THRESHOLD:
                large_files.append((file_path, repo_file_path))
                continue

            operations.append(
                CommitOperationAdd(
                    path_in_repo=repo_file_path,
                    path_or_fileobj=str(file_path),
                )
            )

        # Upload large files as standalone commits
        for file_path, repo_file_path in large_files:
            print(f"\n Large file ({file_path.stat().st_size / (1024**2):.1f} MB): {file_path.name}")
            try:
                api.create_commit(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    operations=[CommitOperationAdd(path_in_repo=repo_file_path, path_or_fileobj=str(file_path))],
                    commit_message=f"Upload large image: {file_path.name}",
                )
                print(f"   Uploaded {file_path.name}")
            except Exception as e:
                print(f"   Failed {file_path.name}: {e}")
                if "429" in str(e):
                    print(" Rate limit reached. Halting script execution.")
                    sys.exit(1)

        # Upload remaining files as a batch
        if operations:
            print(f"\n--- Batch [{idx}/{len(batches)}] ({len(operations)} files, {batch_bytes / (1024**2):.1f} MB) ---")
            try:
                api.create_commit(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    operations=operations,
                    commit_message=f"Upload batch {idx}/{len(batches)}",
                )
                print(f"   Committed batch {idx}")
            except Exception as e:
                print(f"  ✗ Failed batch {idx}: {e}")
                if "429" in str(e):
                    print(" Rate limit reached. Halting script execution.")
                    sys.exit(1)

    print("\nAll uploads complete!")


def main():
    p = argparse.ArgumentParser(description="Upload large image directories to HF Hub in size-capped batches")
    p.add_argument("--dataset-dir", required=True, type=Path, help="Local path to image directory")
    p.add_argument("--repo-id", required=True, help="HF repo ID")
    p.add_argument("--branch", default="main", help="Target branch")
    p.add_argument("--target-mb", type=int, default=150, help="Target batch payload size in MB")
    p.add_argument("--path-in-repo", default="Dataset", help="Path within the repo")
    p.add_argument("--clean-first", action="store_true", help="Delete existing repo files first")

    args = p.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    if args.branch != "main":
        api.create_branch(repo_id=args.repo_id, repo_type="dataset", branch=args.branch, exist_ok=True)

    print(f"Uploading {args.dataset_dir} to {args.repo_id} (branch: {args.branch})...")
    print(f"Target batch size: ~{args.target_mb} MB")

    upload_batch(
        repo_id=args.repo_id,
        local_dir=args.dataset_dir,
        repo_type="dataset",
        revision=args.branch,
        path_in_repo=args.path_in_repo,
        target_mb=args.target_mb,
        clean_first=args.clean_first,
    )


if __name__ == "__main__":
    main()
