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
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    pipeline = TheftDetectionPipeline(
        weights=args.weights,
        zones_path=args.zones,
        conf=args.conf,
        sinks=[LocalLogSink(args.alerts_dir)],
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
