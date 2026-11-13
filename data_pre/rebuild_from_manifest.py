"""Recreate this ImageNet subset from the committed manifests using symlinks."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--source-root", type=Path, required=True, help="ImageNet root with train/ and flat val/")
    parser.add_argument("--output-root", type=Path, help="New directory for the subset")
    parser.add_argument("--verify-only", action="store_true", help="Check manifests and source files without writing")
    args = parser.parse_args()
    if not args.verify_only and args.output_root is None:
        parser.error("--output-root is required unless --verify-only is used")

    manifest_dir = args.manifest_dir.resolve()
    source_root = args.source_root.resolve()
    metadata = json.loads((manifest_dir / "metadata.json").read_text())
    train_list = (manifest_dir / "train_files.txt").read_bytes()
    actual_hash = hashlib.sha256(train_list).hexdigest()
    if actual_hash != metadata["train_file_list_sha256"]:
        raise ValueError("train_files.txt SHA-256 does not match metadata.json")

    rows_by_split = {}
    seen = set()
    for split in ("train", "val"):
        with (manifest_dir / f"{split}_manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            rel = PurePosixPath(row["relative_path"])
            if len(rel.parts) != 3 or rel.parts[0] != split or rel.parts[1] != row["wnid"]:
                raise ValueError(f"Invalid relative path: {rel}")
            if any(part in (".", "..") for part in rel.parts) or row["relative_path"] in seen:
                raise ValueError(f"Unsafe or duplicate path: {rel}")
            if metadata["class_to_idx"][row["wnid"]] != int(row["class_index"]):
                raise ValueError(f"Class index mismatch: {rel}")
            seen.add(row["relative_path"])
            source = source_root / split / (row["wnid"] if split == "train" else "") / rel.name
            if not source.is_file():
                raise FileNotFoundError(source)
            row["local_source"] = source
        rows_by_split[split] = rows

    train_paths = [row["relative_path"] for row in rows_by_split["train"]]
    if train_list.decode("utf-8").splitlines() != train_paths:
        raise ValueError("train_files.txt and train_manifest.csv select different images")
    if len(train_paths) != metadata["train_selected_images"]:
        raise ValueError("Training image count does not match metadata.json")
    if len(rows_by_split["val"]) != metadata["validation_images"]:
        raise ValueError("Validation image count does not match metadata.json")

    if args.verify_only:
        print(f"Verified {len(train_paths)} training and {len(rows_by_split['val'])} validation images")
        return

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    output_root.mkdir(parents=True)
    for split, rows in rows_by_split.items():
        for row in rows:
            destination = output_root / row["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(row["local_source"], destination)
    for name in ("README.md", "metadata.json", "train_manifest.csv", "val_manifest.csv", "train_files.txt", ".gitignore"):
        shutil.copy2(manifest_dir / name, output_root / name)
    print(f"Created {len(train_paths)} training and {len(rows_by_split['val'])} validation links in {output_root}")


if __name__ == "__main__":
    main()
