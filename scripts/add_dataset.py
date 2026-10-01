#!/usr/bin/env python3
"""Download one Roboflow dataset and register it in configs/merge.yaml --
no hand-editing YAML required (that's been the source of repeated syntax
errors when done manually).

Guesses each class's mapping onto unified_classes (normal/suspicious) from
its name via keyword matching, and ALWAYS prints the guess for review --
an unrecognized class name is dropped rather than guessed wrong, so check
the output and configs/merge.yaml before running merge_datasets.py.

Usage:
    python scripts/add_dataset.py --workspace <ws> --project <proj> --version <n>

Requires ROBOFLOW_API_KEY in the environment.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MERGE_CONFIG = ROOT / "configs" / "merge.yaml"

# Keyword heuristics built from every dataset we've actually mapped by hand
# so far (0/1, Normal/Suspicious Behavior, High/Low-suspicion, Shoplifting,
# mencurigakan). An unrecognized class name falls through to None (dropped)
# rather than risk a wrong guess.
NORMAL_HINTS = ("normal", "0")
SUSPICIOUS_HINTS = (
    "suspicious", "shoplift", "theft", "steal", "stealing",
    "high-suspicion", "high_suspicion", "mencurigakan", "1", "2",
)
DROP_HINTS = ("low-suspicion", "low_suspicion", "unlabeled", "unknown")


def guess_target(class_name: str) -> str | None:
    name = class_name.strip().lower()
    if any(h in name for h in DROP_HINTS):
        return None
    if any(name == h or h in name for h in NORMAL_HINTS):
        return "normal"
    if any(h in name for h in SUSPICIOUS_HINTS):
        return "suspicious"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    dest = DATA_DIR / args.project
    if dest.exists():
        print(f"{dest} already exists -- skipping download.")
    else:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            sys.exit("Missing ROBOFLOW_API_KEY env var")
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(args.workspace).project(args.project)
        dataset = proj.version(int(args.version)).download("yolov8", location=str(dest))
        print(f"Downloaded to {dataset.location}")

    data_yaml = dest / "data.yaml"
    if not data_yaml.exists():
        matches = list(dest.glob("**/data.yaml"))
        if not matches:
            sys.exit(f"No data.yaml found under {dest} -- the download may have failed partway.")
        data_yaml = matches[0]

    with open(data_yaml) as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    class_map = {}
    print("\nGuessed class mapping (REVIEW THIS):")
    for name in names:
        target = guess_target(name)
        print(f"  '{name}' -> {target!r}")
        if target:
            class_map[name] = target

    if MERGE_CONFIG.exists():
        with open(MERGE_CONFIG) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"unified_classes": ["normal", "suspicious"], "sources": []}

    rel_path = f"data/raw/{args.project}"
    cfg["sources"] = [s for s in cfg["sources"] if s["path"] != rel_path]
    cfg["sources"].append({"path": rel_path, "class_map": class_map})

    with open(MERGE_CONFIG, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"\nWrote {MERGE_CONFIG} ({len(cfg['sources'])} sources total).")
    if any(guess_target(n) is None for n in names):
        print("NOTE: at least one class had no confident guess and was dropped -- "
              "open configs/merge.yaml and fix it by hand if that class matters.")


if __name__ == "__main__":
    main()
