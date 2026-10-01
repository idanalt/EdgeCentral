"""Wires together: YOLO detection+tracking -> behavior heuristics -> alert sinks."""
from __future__ import annotations

import time

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from detection.alerts import AlertSink
from detection.behavior import ConcealmentDetector, Zone


def load_zones(path: str | None) -> list[Zone]:
    if not path:
        return []
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return [Zone.from_dict(z) for z in cfg.get("zones", [])]


class TheftDetectionPipeline:
    def __init__(
        self,
        weights: str,
        zones_path: str | None = None,
        conf: float = 0.35,
        sinks: list[AlertSink] | None = None,
        person_classes: set[str] | None = None,
        item_classes: set[str] | None = None,
        suspicious_classes: set[str] | None = None,
        min_suspicious_conf: float = 0.6,
    ) -> None:
        self.model = YOLO(weights)
        self.conf = conf
        self.detector = ConcealmentDetector(
            zones=load_zones(zones_path),
            person_classes=person_classes,
            item_classes=item_classes,
            suspicious_classes=suspicious_classes,
            min_suspicious_conf=min_suspicious_conf,
        )
        self.sinks = sinks or []

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        result = self.model.track(frame, conf=self.conf, persist=True, verbose=False)[0]
        detections = []
        if result.boxes is not None and result.boxes.id is not None:
            names = result.names
            for box, track_id, cls_idx, conf in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.id.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            ):
                detections.append(
                    {
                        "track_id": int(track_id),
                        "cls": names[int(cls_idx)],
                        "bbox": tuple(box.tolist()),
                        "conf": float(conf),
                    }
                )

        alerts = self.detector.update(detections, now=time.time())
        for alert in alerts:
            for sink in self.sinks:
                sink.emit(alert, frame=frame)

        annotated = result.plot()
        for alert in alerts:
            x1, y1, x2, y2 = map(int, alert.bbox)
            color = {"low": (0, 200, 200), "medium": (0, 140, 255), "high": (0, 0, 255)}[alert.severity]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                annotated,
                alert.kind,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return annotated, alerts
