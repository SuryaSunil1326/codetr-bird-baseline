#!/usr/bin/env python3
"""
upload_dataset.py

Uploads large datasets to Hugging Face Hub using api.upload_folder.
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


def main():
    p = argparse.ArgumentParser(description="Upload dataset folder to HF Hub")
    p.add_argument("--dataset-dir", required=True, type=Path, help="Local directory")
    p.add_argument("--repo-id", required=True, help="HF Repo ID")
    p.add_argument("--branch", default="dev", help="Target branch")
    p.add_argument("--path-in-repo", default="Dataset", help="Path inside repo")
    p.add_argument("--clean-first", action="store_false", help="Clean stray root files first")
    args = p.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    if args.branch != "main":
        api.create_branch(repo_id=args.repo_id, repo_type="dataset", branch=args.branch, exist_ok=True)

    if args.clean_first:
        clean_stray_root_files(api, args.repo_id, "dataset", args.branch)

    print(f"🚀 Uploading {args.dataset_dir} -> {args.repo_id} (branch: {args.branch})...")

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
