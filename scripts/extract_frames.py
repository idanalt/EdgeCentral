#!/usr/bin/env python3
"""Extract frames from a video file for manual labeling.

Training always happens on individual labeled images, never raw video --
this just turns a video into a batch of candidate frames to upload and
label in Roboflow (or any other YOLO-format labeling tool).
"""
import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="directory to save extracted frames into")
    parser.add_argument(
        "--every-seconds",
        type=float,
        default=0.5,
        help="save one frame every this many seconds of video (default: 0.5)",
    )
    parser.add_argument("--prefix", default=None, help="filename prefix (default: video's stem)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.video.stem

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(fps * args.every_seconds))

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            out_path = args.out / f"{prefix}_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {args.out} (from {frame_idx} total frames, every {args.every_seconds}s)")


if __name__ == "__main__":
    main()
