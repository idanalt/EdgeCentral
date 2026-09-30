from detection.behavior import ConcealmentDetector, Zone


def test_no_alert_for_normal_browsing():
    det = ConcealmentDetector()
    t = 0.0
    # Person and item stay visible together for several frames -> no concealment.
    for _ in range(10):
        alerts = det.update(
            [
                {"track_id": 1, "cls": "person", "bbox": (100, 100, 200, 300)},
                {"track_id": 2, "cls": "item", "bbox": (140, 180, 160, 220)},
            ],
            now=t,
        )
        assert alerts == []
        t += 0.2


def test_concealment_alert_when_item_vanishes_near_person():
    det = ConcealmentDetector()
    t = 0.0
    # Item overlaps the person's torso region.
    det.update(
        [
            {"track_id": 1, "cls": "person", "bbox": (100, 100, 200, 300)},
            {"track_id": 2, "cls": "item", "bbox": (140, 180, 160, 220)},
        ],
        now=t,
    )
    t += 0.2

    # Item disappears; person keeps being tracked past the grace period.
    alerts = []
    for _ in range(10):
        t += 0.5
        alerts = det.update(
            [{"track_id": 1, "cls": "person", "bbox": (105, 100, 205, 300)}],
            now=t,
        )
        if alerts:
            break

    assert len(alerts) == 1
    assert alerts[0].kind == "concealment"
    assert alerts[0].track_id == 1


def test_no_concealment_if_person_also_disappears():
    det = ConcealmentDetector()
    det.update(
        [
            {"track_id": 1, "cls": "person", "bbox": (100, 100, 200, 300)},
            {"track_id": 2, "cls": "item", "bbox": (140, 180, 160, 220)},
        ],
        now=0.0,
    )
    # Both vanish together (person just walked off-screen normally).
    alerts = det.update([], now=5.0)
    assert alerts == []


def test_loitering_alert_after_dwell_threshold():
    shelf = Zone(
        name="shelf1",
        type="shelf",
        polygon=__import__("shapely.geometry", fromlist=["Polygon"]).Polygon(
            [(0, 0), (300, 0), (300, 300), (0, 300)]
        ),
        dwell_alert_seconds=2.0,
    )
    det = ConcealmentDetector(zones=[shelf])
    t = 0.0
    alerts = det.update([{"track_id": 1, "cls": "person", "bbox": (100, 100, 150, 200)}], now=t)
    assert alerts == []

    t += 3.0
    alerts = det.update([{"track_id": 1, "cls": "person", "bbox": (100, 100, 150, 200)}], now=t)
    assert len(alerts) == 1
    assert alerts[0].kind == "loitering"


def test_exit_with_concealment_is_high_priority():
    from shapely.geometry import Polygon

    exit_zone = Zone(name="exit1", type="exit", polygon=Polygon([(500, 0), (800, 0), (800, 300), (500, 300)]))
    det = ConcealmentDetector(zones=[exit_zone])

    # Manually flag the track as concealed (simulating a prior concealment alert).
    det.update([{"track_id": 1, "cls": "person", "bbox": (0, 0, 50, 50)}], now=0.0)
    det._tracks[1].concealment_flag_until = 999999.0

    alerts = det.update([{"track_id": 1, "cls": "person", "bbox": (600, 100, 650, 150)}], now=1.0)
    assert any(a.kind == "exit_with_concealment" and a.severity == "high" for a in alerts)
