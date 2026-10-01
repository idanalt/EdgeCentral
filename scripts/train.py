#!/usr/bin/env python3
"""Train the YOLO theft/suspicious-behavior detector.

Reads defaults from configs/train.yaml; any CLI flag overrides the
corresponding YAML key. Requires a GPU for realistic training times —
run this on a workstation/cloud GPU, then copy runs/detect/*/weights/best.pt
to the edge device for export/inference.
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = ROOT / "configs" / "train.yaml"


def _str_to_bool(s: str) -> bool:
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {s!r}")


def main() -> None:
    with open(DEFAULT_CFG) as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    for key, val in cfg.items():
        # bool is a subclass of int, so `type=type(val)` would otherwise bind
        # `bool`, and argparse's default str->bool coercion treats any
        # non-empty string (including "False") as truthy.
        arg_type = _str_to_bool if isinstance(val, bool) else type(val) if val is not None else str
        parser.add_argument(f"--{key.replace('_', '-')}", default=val, type=arg_type)
    parser.add_argument("--resume", action="store_true", help="resume the last interrupted run")
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "resume"}
    data_path = overrides.pop("data")
    model_path = overrides.pop("model")

    if not (ROOT / data_path).exists() and not Path(data_path).is_absolute():
        raise SystemExit(
            f"{data_path} not found — run scripts/download_dataset.py and "
            "scripts/prepare_dataset.py first."
        )

    model = YOLO(model_path)
    model.train(data=data_path, resume=args.resume, **overrides)


if __name__ == "__main__":
    main()
