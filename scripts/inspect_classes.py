#!/usr/bin/env python3
"""Save a handful of example images per class, with the labeled box drawn,
so a human can tell what a generically-named class (e.g. "0"/"1" instead of
"Normal"/"Suspicious") actually represents before wiring it into the
behavior heuristics.
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "configs" / "dataset.yaml"))
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--out", default=str(ROOT / "data" / "inspect"))
    args = parser.parse_args()

    with open(args.dataset) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    dataset_root = Path(cfg["path"])
    images_dir = dataset_root / args.split / "images"
    labels_dir = dataset_root / args.split / "labels"

    label_files = list(labels_dir.glob("*.txt"))
    random.shuffle(label_files)

    by_class: dict[int, list[Path]] = defaultdict(list)
    n_classes = len(names)
    for lf in label_files:
        lines = [ln.strip() for ln in lf.read_text().splitlines() if ln.strip()]
        for c in {int(ln.split()[0]) for ln in lines}:
            if len(by_class[c]) < args.per_class:
                by_class[c].append(lf)
        if len(by_class) == n_classes and all(len(v) >= args.per_class for v in by_class.values()):
            break

    out_root = Path(args.out)
    for cls_idx, cls_label_files in by_class.items():
        cls_name = names[cls_idx] if isinstance(names, list) else names[cls_idx]
        out_dir = out_root / f"class_{cls_idx}_{cls_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for lf in cls_label_files:
            img_path = next(
                (images_dir / f"{lf.stem}{ext}" for ext in (".jpg", ".jpeg", ".png") if (images_dir / f"{lf.stem}{ext}").exists()),
                None,
            )
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            for line in lf.read_text().splitlines():
                parts = line.split()
                if not parts or int(parts[0]) != cls_idx:
                    continue
                xc, yc, bw, bh = map(float, parts[1:5])
                x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
                x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(str(out_dir / img_path.name), img)

    print(f"Saved sample images under {out_root} — open each class_* folder and look at what the box shows.")


if __name__ == "__main__":
    main()
