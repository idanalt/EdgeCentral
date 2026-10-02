#!/usr/bin/env python3
"""Turn a folder of self-captured images into a labeled YOLO dataset --
no manual bounding-box drawing (Roboflow UI etc.) needed.

Use this only when each image has a single dominant subject filling most
of the frame (like a webcam torso shot) -- it draws one box per image
covering a configurable fraction of the frame, centered. For images with
multiple people, small/distant subjects, or background clutter near the
subject, use Roboflow's proper labeling tool instead -- a wrong box here
is worse than no data.

Input layout -- sort images into one subfolder per class (the subfolder
name becomes the class name directly, so name them to match your
unified_classes, e.g. "normal"/"suspicious"):

    <input>/normal/*.jpg
    <input>/suspicious/*.jpg

Output: a standard YOLO dataset (train/valid split, data.yaml) under
data/raw/<name>, ready to add to configs/merge.yaml like any other source.
"""
import argparse
import random
import shutil
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="folder with one subfolder per class")
    parser.add_argument("--out", required=True, type=Path, help="output dataset folder, e.g. data/raw/my_custom")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--box-width-frac", type=float, default=0.85, help="box width as a fraction of image width")
    parser.add_argument("--box-height-frac", type=float, default=0.95, help="box height as a fraction of image height")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    class_dirs = sorted(p for p in args.input.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class subfolders found under {args.input}")
    class_names = [p.name for p in class_dirs]
    print(f"Classes: {class_names}")

    random.seed(args.seed)

    # YOLO box: class cx cy w h, all normalized 0-1. Centered box covering
    # the given fraction of the frame.
    cx, cy = 0.5, 0.5
    bw, bh = args.box_width_frac, args.box_height_frac
    label_line = None  # set per class below

    counts = {"train": 0, "valid": 0}
    for cls_idx, class_dir in enumerate(class_dirs):
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        random.shuffle(images)
        n_val = max(1, round(len(images) * args.val_split)) if len(images) > 1 else 0
        splits = {"valid": images[:n_val], "train": images[n_val:]}

        label_line = f"{cls_idx} {cx} {cy} {bw} {bh}"
        for split, split_images in splits.items():
            img_out = args.out / split / "images"
            lbl_out = args.out / split / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, img_out / img_path.name)
                (lbl_out / f"{img_path.stem}.txt").write_text(label_line + "\n")
                counts[split] += 1

    data_yaml = {
        "path": str(args.out.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    with open(args.out / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"train: {counts['train']} images, valid: {counts['valid']} images")
    print(f"Wrote {args.out}/data.yaml")
    print(
        f"\nAdd to configs/merge.yaml:\n"
        f"  - path: {args.out}\n"
        f"    class_map:\n" + "\n".join(f'      "{n}": {n}' for n in class_names)
    )


if __name__ == "__main__":
    main()
