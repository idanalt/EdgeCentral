#!/usr/bin/env python3
"""Merge several YOLO-format datasets into one unified dataset.

Why: a model trained on a single store's footage overfits to that scene
(camera angle, lighting, layout). Deploying to many different sites needs
one model trained on *diverse* data -- multiple public datasets plus
footage from several of your own sites, merged together. Each source may
use different class names/indices; configs/merge.example.yaml shows how
to map them onto one shared class list.
"""
import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_source_names(source_root: Path) -> tuple[dict[int, str], Path]:
    data_yaml = source_root / "data.yaml"
    if not data_yaml.exists():
        matches = list(source_root.glob("**/data.yaml"))
        if not matches:
            raise SystemExit(f"No data.yaml found under {source_root}")
        data_yaml = matches[0]
        source_root = data_yaml.parent
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(i): n for i, n in names.items()}, source_root


def build_remap(source_names: dict[int, str], class_map: dict, unified_classes: list[str]) -> dict[int, int | None]:
    """original class index -> unified class index, or None to drop it."""
    class_map_norm = {str(k): v for k, v in class_map.items()}
    unified_index = {name: i for i, name in enumerate(unified_classes)}

    remap: dict[int, int | None] = {}
    for idx, name in source_names.items():
        target = class_map_norm.get(name, class_map_norm.get(str(idx)))
        if target is None:
            print(f"  WARNING: class '{name}' (idx {idx}) not in class_map -- its labels will be dropped")
            remap[idx] = None
            continue
        if target not in unified_index:
            raise SystemExit(f"class_map target '{target}' is not in unified_classes {unified_classes}")
        remap[idx] = unified_index[target]
    return remap


def merge_split(source_root: Path, split: str, slug: str, remap: dict, out_root: Path) -> tuple[int, int]:
    images_dir = source_root / split / "images"
    labels_dir = source_root / split / "labels"
    if not images_dir.is_dir():
        return 0, 0

    out_images = out_root / split / "images"
    out_labels = out_root / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = 0
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"

        lines_out = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                new_cls = remap.get(int(parts[0]))
                if new_cls is None:
                    continue
                lines_out.append(" ".join([str(new_cls), *parts[1:]]))

        # Prefix with the source slug so filenames never collide across datasets.
        shutil.copy2(img_path, out_images / f"{slug}__{img_path.name}")
        (out_labels / f"{slug}__{img_path.stem}.txt").write_text("\n".join(lines_out))
        n_images += 1
        n_boxes += len(lines_out)

    return n_images, n_boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "merge.yaml"))
    parser.add_argument("--out", default=str(ROOT / "data" / "merged"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    unified_classes = cfg["unified_classes"]

    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)

    totals = {"train": [0, 0], "valid": [0, 0], "test": [0, 0]}
    for source in cfg["sources"]:
        source_root = (ROOT / source["path"]).resolve()
        slug = Path(source["path"]).name
        print(f"Source: {source_root} (slug={slug})")

        source_names, resolved_root = load_source_names(source_root)
        print(f"  classes: {source_names}")
        remap = build_remap(source_names, source["class_map"], unified_classes)

        for split in ("train", "valid", "test"):
            n_img, n_box = merge_split(resolved_root, split, slug, remap, out_root)
            if n_img:
                totals[split][0] += n_img
                totals[split][1] += n_box
                print(f"  {split}: {n_img} images, {n_box} boxes merged")

    data_yaml_out = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "names": {i: n for i, n in enumerate(unified_classes)},
    }
    if totals["test"][0]:
        data_yaml_out["test"] = "test/images"

    out_config = ROOT / "configs" / "dataset.yaml"
    with open(out_config, "w") as f:
        yaml.safe_dump(data_yaml_out, f, sort_keys=False, allow_unicode=True)

    print("\n--- Totals ---")
    for split, (n_img, n_box) in totals.items():
        if n_img:
            print(f"{split}: {n_img} images, {n_box} boxes")
    print(f"\nWrote merged dataset to {out_root}")
    print(f"Wrote {out_config}")


if __name__ == "__main__":
    main()
