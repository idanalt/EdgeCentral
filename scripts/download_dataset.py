#!/usr/bin/env python3
"""Download a labeled YOLO-format dataset from Roboflow.

Requires a free Roboflow account and API key. Point this at a public
"shoplifting detection" / "theft detection" dataset from Roboflow Universe,
or your own private annotated project.

Env vars required:
    ROBOFLOW_API_KEY
    ROBOFLOW_WORKSPACE
    ROBOFLOW_PROJECT
    ROBOFLOW_VERSION
"""
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    workspace = os.environ.get("ROBOFLOW_WORKSPACE")
    project = os.environ.get("ROBOFLOW_PROJECT")
    version = os.environ.get("ROBOFLOW_VERSION")

    missing = [
        name
        for name, val in [
            ("ROBOFLOW_API_KEY", api_key),
            ("ROBOFLOW_WORKSPACE", workspace),
            ("ROBOFLOW_PROJECT", project),
            ("ROBOFLOW_VERSION", version),
        ]
        if not val
    ]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        print(
            "See README.md 'שלב 1: דאטהסט' for how to obtain a Roboflow API key "
            "and pick a shoplifting/theft-detection dataset from Roboflow Universe.",
            file=sys.stderr,
        )
        sys.exit(1)

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace)
    proj = ws.project(project)
    dataset = proj.version(int(version)).download("yolov8", location=str(DATA_DIR))
    print(f"Downloaded dataset to: {dataset.location}")


if __name__ == "__main__":
    main()
