#!/usr/bin/env python3
"""Run validation metrics (mAP, precision/recall, confusion matrix) for a
trained checkpoint against the held-out val/test split."""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", default=str(ROOT / "configs" / "dataset.yaml"))
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    metrics = model.val(data=args.data, split=args.split, conf=args.conf)

    print("\n--- Summary ---")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(
        "\nFull confusion matrix / per-class plots saved under the run's "
        "'val' directory (see path printed above by Ultralytics)."
    )


if __name__ == "__main__":
    main()
