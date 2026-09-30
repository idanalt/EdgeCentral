#!/usr/bin/env python3
"""Validate a downloaded YOLO dataset and produce the final configs/dataset.yaml.

Checks that each split (train/valid/test) has matching images/ and labels/
directories, reports per-class instance counts (useful for spotting a
severely imbalanced "shoplifting" vs "normal" dataset before training),
and writes out configs/dataset.yaml pointing at the validated data.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "data" / "raw"
OUT_CONFIG = ROOT / "configs" / "dataset.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_source_yaml(raw_dir: Path) -> Path:
    candidates = list(raw_dir.glob("**/data.yaml"))
    if not candidates:
        print(f"No data.yaml found under {raw_dir}. Run scripts/download_dataset.py first.", file=sys.stderr)
        sys.exit(1)
    # Prefer the shallowest match.
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def validate_split(split_dir: Path, class_names: list[str]) -> Counter:
    counts: Counter = Counter()
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return counts

    images = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem for p in labels_dir.glob("*.txt")}

    missing_labels = images - labels
    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} images in {images_dir} have no label file")

    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls_idx = int(line.split()[0])
            if 0 <= cls_idx < len(class_names):
                counts[class_names[cls_idx]] += 1
            else:
                print(f"  WARNING: class index {cls_idx} out of range in {label_file}")

    print(f"  {split_dir.name}: {len(images)} images, {len(labels)} label files")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()

    src_yaml = find_source_yaml(args.raw_dir)
    dataset_root = src_yaml.parent
    with open(src_yaml) as f:
        src_cfg = yaml.safe_load(f)

    class_names = src_cfg["names"]
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names)]

    print(f"Source dataset: {dataset_root}")
    print(f"Classes ({len(class_names)}): {class_names}")

    total_counts: Counter = Counter()
    for split in ("train", "valid", "test"):
        split_dir = dataset_root / split
        if split_dir.is_dir():
            total_counts.update(validate_split(split_dir, class_names))

    if not total_counts:
        print("No labeled instances found anywhere — dataset looks empty or malformed.", file=sys.stderr)
        sys.exit(1)

    print("\nInstance counts per class:")
    for name, count in total_counts.most_common():
        print(f"  {name}: {count}")

    out_cfg = {
        "path": str(dataset_root),
        "train": "train/images",
        "val": "valid/images" if (dataset_root / "valid").is_dir() else "val/images",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    if (dataset_root / "test").is_dir():
        out_cfg["test"] = "test/images"

    OUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CONFIG, "w") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False, allow_unicode=True)
    print(f"\nWrote {OUT_CONFIG}")


if __name__ == "__main__":
    main()
