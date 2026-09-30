"""Alert sinks — where behavior alerts go once raised.

Every sink here is meant to notify a human for review (a staff dashboard,
a log a security manager checks, a webhook to an internal tool). None of
them take automated action against a person. Do not wire these into
automated doors, law-enforcement dispatch, or any system that acts on a
person without human review — see README.md.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from detection.behavior import Alert


class AlertSink(ABC):
    @abstractmethod
    def emit(self, alert: Alert, frame: np.ndarray | None = None) -> None: ...


class LocalLogSink(AlertSink):
    """Writes each alert as a JSON line plus an optional snapshot image,
    for a staff member to review later — never acted on automatically."""

    def __init__(self, out_dir: str | Path = "alerts") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "events.jsonl"

    def emit(self, alert: Alert, frame: np.ndarray | None = None) -> None:
        ts = datetime.fromtimestamp(alert.timestamp, tz=timezone.utc).isoformat()
        record = {
            "timestamp": ts,
            "kind": alert.kind,
            "track_id": alert.track_id,
            "severity": alert.severity,
            "score": alert.score,
            "message": alert.message,
            "bbox": alert.bbox,
        }

        snapshot_path = None
        if frame is not None:
            snapshot_name = f"{alert.kind}_{alert.track_id}_{int(alert.timestamp)}.jpg"
            snapshot_path = self.out_dir / snapshot_name
            cv2.imwrite(str(snapshot_path), frame)
            record["snapshot"] = str(snapshot_path)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


class WebhookSink(AlertSink):
    """POSTs alerts to an internal staff-review endpoint (e.g. a dashboard
    or ticketing system). Requires `requests`."""

    def __init__(self, url: str, timeout: float = 3.0) -> None:
        self.url = url
        self.timeout = timeout

    def emit(self, alert: Alert, frame: np.ndarray | None = None) -> None:
        import requests

        payload = {
            "timestamp": alert.timestamp,
            "kind": alert.kind,
            "track_id": alert.track_id,
            "severity": alert.severity,
            "score": alert.score,
            "message": alert.message,
            "bbox": alert.bbox,
        }
        try:
            requests.post(self.url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            print(f"WebhookSink: failed to send alert: {exc}")
