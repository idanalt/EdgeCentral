#!/usr/bin/env python3
"""Real-time entrypoint: run detection+tracking+behavior alerts against a
webcam index, video file, or RTSP stream, saving alert snapshots for
staff review. Not an automated action system — see README.md.

Usage:
    python -m inference.run_stream --weights best.onnx --source 0
    python -m inference.run_stream --weights best.pt --source rtsp://cam/stream --zones configs/zones.yaml
"""
import argparse

import cv2

from detection.alerts import LocalLogSink
from detection.pipeline import TheftDetectionPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="webcam index, video path, or RTSP URL")
    parser.add_argument("--zones", default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--alerts-dir", default="alerts")
    parser.add_argument("--display", action="store_true", help="show a live preview window")
    parser.add_argument(
        "--person-classes",
        default="person",
        help="comma-separated class names that represent a trackable person (used for zone dwell/exit logic)",
    )
    parser.add_argument(
        "--item-classes",
        default="",
        help="comma-separated class names that can be concealed (leave empty if the dataset has no separate item class)",
    )
    parser.add_argument(
        "--suspicious-classes",
        default="",
        help="comma-separated class names that directly label suspicious behavior on a single box "
        "(e.g. 'Suspicious Behavior' for a dataset with Normal/Suspicious Behavior classes)",
    )
    parser.add_argument(
        "--min-suspicious-conf",
        type=float,
        default=0.6,
        help="a frame only counts toward the sustained-suspicious streak if the model's confidence "
        "for it is at least this -- raise it if normal item-holding is triggering alerts",
    )
    args = parser.parse_args()

    def _split(s: str) -> set[str]:
        return {c.strip() for c in s.split(",") if c.strip()}

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    pipeline = TheftDetectionPipeline(
        weights=args.weights,
        zones_path=args.zones,
        conf=args.conf,
        sinks=[LocalLogSink(args.alerts_dir)],
        person_classes=_split(args.person_classes),
        item_classes=_split(args.item_classes) or None,
        suspicious_classes=_split(args.suspicious_classes) or None,
        min_suspicious_conf=args.min_suspicious_conf,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            annotated, alerts = pipeline.process_frame(frame)
            for alert in alerts:
                print(f"[{alert.severity.upper()}] {alert.kind}: {alert.message}")

            if args.display:
                cv2.imshow("theft-detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
