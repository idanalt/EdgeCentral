"""Behavior heuristics on top of raw YOLO+tracking detections.

Object detection alone cannot tell you "this person is stealing" — theft is
a *behavior* over time, not a single bounding box. This module turns a
stream of per-frame tracked detections into scored alerts using simple,
auditable heuristics:

  1. Concealment: an "item"-class track that was overlapping a person's
     torso region disappears while that person keeps moving (classic
     "into the bag/jacket" signal). Used with datasets that detect
     persons and items as separate classes.
  2. Sustained suspicious behavior: a track classified directly as a
     "suspicious" class (e.g. a dataset that labels the whole behavior
     as one box, like "Suspicious Behavior" vs "Normal Behavior") holds
     that class for several consecutive seconds — filters out one-frame
     detector noise before alerting.
  3. Loitering: a person dwells inside a configured "shelf" zone longer
     than a threshold.
  4. Exit-with-concealment: a person carrying an active concealment or
     sustained-suspicious flag enters a configured "exit" zone — the
     highest-priority alert.

These are heuristics, not proof. Every alert exists to be reviewed by a
human before any action is taken.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from shapely.geometry import Point, Polygon

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2

DEFAULT_PERSON_CLASSES = {"person"}
DEFAULT_ITEM_CLASSES = {"item", "product", "bag", "backpack", "handbag", "bottle"}

# How long an item can go untracked (occlusion, brief detector miss) before
# we treat it as "disappeared" rather than "briefly lost".
ITEM_DISAPPEAR_GRACE_SECONDS = 1.5
# How long a concealment/sustained-suspicious flag stays valid on a person
# track before decaying (long enough to survive a short walk to the exit).
CONCEALMENT_FLAG_TTL_SECONDS = 60.0
# How long a track must hold a "suspicious" class continuously before it
# counts as a real signal rather than one noisy frame.
SUSPICIOUS_SUSTAIN_SECONDS = 1.5
# A brief gap (missed frame, momentary re-classification) doesn't reset the
# streak; a gap longer than this does.
SUSPICIOUS_RESET_GRACE_SECONDS = 1.0


def _center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _torso_region(person_bbox: BBox) -> BBox:
    """Rough torso/pocket sub-region of a person's bbox (middle third, vertically)."""
    x1, y1, x2, y2 = person_bbox
    h = y2 - y1
    return x1, y1 + h * 0.25, x2, y1 + h * 0.75


def _contains(outer: BBox, point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = outer
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


@dataclass
class Zone:
    name: str
    type: str  # "shelf" | "exit"
    polygon: Polygon
    dwell_alert_seconds: float = 8.0

    @classmethod
    def from_dict(cls, d: dict) -> "Zone":
        return cls(
            name=d["name"],
            type=d["type"],
            polygon=Polygon(d["polygon"]),
            dwell_alert_seconds=d.get("dwell_alert_seconds", 8.0),
        )


@dataclass
class TrackState:
    track_id: int
    cls: str
    last_bbox: BBox
    last_seen: float
    first_seen: float
    concealment_flag_until: float = 0.0
    zone_enter_time: dict[str, float] = field(default_factory=dict)
    loiter_alerted: set[str] = field(default_factory=set)
    suspicious_since: float | None = None
    suspicious_last_seen: float | None = None
    suspicious_alerted: bool = False
    suspicious_max_conf: float = 0.0

    def is_flagged(self, now: float) -> bool:
        return self.concealment_flag_until > now


@dataclass
class Alert:
    kind: str  # "concealment" | "suspicious_behavior" | "loitering" | "exit_with_concealment"
    track_id: int
    severity: str  # "low" | "medium" | "high"
    score: float
    message: str
    bbox: BBox
    timestamp: float
    model_conf: float | None = None  # the YOLO classification confidence that triggered this, if applicable


class ConcealmentDetector:
    """Stateful detector: feed it tracked detections frame by frame."""

    def __init__(
        self,
        zones: list[Zone] | None = None,
        person_classes: set[str] | None = None,
        item_classes: set[str] | None = None,
        suspicious_classes: set[str] | None = None,
        min_suspicious_conf: float = 0.0,
    ) -> None:
        self.zones = zones or []
        self.person_classes = person_classes or DEFAULT_PERSON_CLASSES
        self.item_classes = item_classes or DEFAULT_ITEM_CLASSES
        # Classes that directly label a box as suspicious behavior (single-
        # class-per-box datasets, e.g. "Suspicious Behavior" vs "Normal
        # Behavior") rather than requiring separate person+item boxes.
        self.suspicious_classes = suspicious_classes or set()
        # A frame's classification only counts toward the sustained-suspicious
        # streak if the model's own confidence for it clears this bar --
        # borderline frames (e.g. 0.4-0.6) are exactly where "holding an item
        # normally" gets misread as concealment, so this filters them out.
        self.min_suspicious_conf = min_suspicious_conf
        self._tracks: dict[int, TrackState] = {}
        # item_track_id -> the person_track_id it was last seen near
        self._item_last_person: dict[int, int] = {}

    def update(self, detections: list[dict], now: float | None = None) -> list[Alert]:
        """detections: [{track_id, cls, bbox, conf}, ...] for the current frame."""
        now = now if now is not None else time.time()
        alerts: list[Alert] = []
        seen_ids = set()

        persons = [d for d in detections if d["cls"] in self.person_classes]
        items = [d for d in detections if d["cls"] in self.item_classes]

        for d in detections:
            seen_ids.add(d["track_id"])
            state = self._tracks.get(d["track_id"])
            if state is None:
                state = TrackState(
                    track_id=d["track_id"],
                    cls=d["cls"],
                    last_bbox=d["bbox"],
                    last_seen=now,
                    first_seen=now,
                )
                self._tracks[d["track_id"]] = state
            else:
                state.last_bbox = d["bbox"]
                state.last_seen = now

        # 1. Associate items with the nearest overlapping person's torso region.
        for item in items:
            item_center = _center(item["bbox"])
            for person in persons:
                if _contains(_torso_region(person["bbox"]), item_center):
                    self._item_last_person[item["track_id"]] = person["track_id"]
                    break

        # 2. Detect concealment: an associated item track vanished.
        for item_id, person_id in list(self._item_last_person.items()):
            item_state = self._tracks.get(item_id)
            person_state = self._tracks.get(person_id)
            if item_state is None or person_state is None:
                continue
            item_gone = item_id not in seen_ids and (now - item_state.last_seen) > ITEM_DISAPPEAR_GRACE_SECONDS
            person_still_present = person_id in seen_ids
            if item_gone and person_still_present:
                person_state.concealment_flag_until = now + CONCEALMENT_FLAG_TTL_SECONDS
                alerts.append(
                    Alert(
                        kind="concealment",
                        track_id=person_id,
                        severity="medium",
                        score=0.6,
                        message=(
                            f"Item (track {item_id}, class={item_state.cls}) disappeared near "
                            f"person track {person_id} — possible concealment."
                        ),
                        bbox=person_state.last_bbox,
                        timestamp=now,
                    )
                )
                del self._item_last_person[item_id]

        # 3. Sustained suspicious-behavior classes (direct single-box labeling).
        #
        # Streak continuity is based on the CLASS alone, not per-frame
        # confidence -- confidence naturally jitters frame to frame even for
        # a genuine event (e.g. hovering right at 0.55-0.6), and gating every
        # single frame on it meant a real sustained concealment could never
        # accumulate the full window if any one frame dipped below the bar.
        # Instead we track the max confidence seen during the streak and only
        # check it against the bar once, at the moment we'd otherwise alert.
        for d in detections:
            state = self._tracks[d["track_id"]]
            is_suspicious_class = d["cls"] in self.suspicious_classes
            if is_suspicious_class:
                if (
                    state.suspicious_since is None
                    or state.suspicious_last_seen is None
                    or now - state.suspicious_last_seen > SUSPICIOUS_RESET_GRACE_SECONDS
                ):
                    state.suspicious_since = now
                    state.suspicious_alerted = False
                    state.suspicious_max_conf = 0.0
                state.suspicious_last_seen = now
                state.suspicious_max_conf = max(state.suspicious_max_conf, d.get("conf", 1.0))

                sustained = now - state.suspicious_since
                if (
                    sustained >= SUSPICIOUS_SUSTAIN_SECONDS
                    and not state.suspicious_alerted
                    and state.suspicious_max_conf >= self.min_suspicious_conf
                ):
                    state.suspicious_alerted = True
                    state.concealment_flag_until = now + CONCEALMENT_FLAG_TTL_SECONDS
                    alerts.append(
                        Alert(
                            kind="suspicious_behavior",
                            track_id=d["track_id"],
                            severity="medium",
                            score=min(1.0, sustained / (SUSPICIOUS_SUSTAIN_SECONDS * 2)),
                            message=(
                                f"Person track {d['track_id']} classified as '{d['cls']}' for "
                                f"{sustained:.1f}s continuously."
                            ),
                            bbox=d["bbox"],
                            timestamp=now,
                            model_conf=state.suspicious_max_conf,
                        )
                    )
            else:
                # Seeing a non-suspicious class for this track resets the streak
                # (it will re-arm for grace-period-tolerant re-triggering above).
                state.suspicious_since = None
                state.suspicious_last_seen = None
                state.suspicious_max_conf = 0.0

        # 4. Zone-based heuristics for persons.
        for person in persons:
            state = self._tracks[person["track_id"]]
            point = Point(_center(person["bbox"]))
            for zone in self.zones:
                inside = zone.polygon.contains(point)
                if inside:
                    if zone.name not in state.zone_enter_time:
                        state.zone_enter_time[zone.name] = now
                    dwell = now - state.zone_enter_time[zone.name]

                    if zone.type == "shelf" and dwell >= zone.dwell_alert_seconds and zone.name not in state.loiter_alerted:
                        state.loiter_alerted.add(zone.name)
                        alerts.append(
                            Alert(
                                kind="loitering",
                                track_id=person["track_id"],
                                severity="low",
                                score=0.3,
                                message=f"Person track {person['track_id']} lingered {dwell:.1f}s near '{zone.name}'.",
                                bbox=person["bbox"],
                                timestamp=now,
                            )
                        )

                    if zone.type == "exit" and state.is_flagged(now):
                        alerts.append(
                            Alert(
                                kind="exit_with_concealment",
                                track_id=person["track_id"],
                                severity="high",
                                score=0.85,
                                message=(
                                    f"Person track {person['track_id']} reached exit zone "
                                    f"'{zone.name}' with an active concealment flag."
                                ),
                                bbox=person["bbox"],
                                timestamp=now,
                            )
                        )
                        state.concealment_flag_until = 0.0  # consume the flag
                else:
                    state.zone_enter_time.pop(zone.name, None)

        # Drop stale tracks to bound memory.
        stale_ids = [tid for tid, s in self._tracks.items() if now - s.last_seen > 300]
        for tid in stale_ids:
            del self._tracks[tid]

        return alerts
