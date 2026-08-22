"""Tests for the duplicate-plate fix: one stopped vehicle, one read.

The bug these pin down: a car parked inside the ALPR zone was recorded again
and again -- a new row every few minutes, each with a different OCR guess at the
same plate -- because the tracker kept renumbering the motionless vehicle and
every new track id started a fresh attempt budget and a fresh database row.

Run from the backend/ directory:  python -m tests.test_vehicle_registry
(No heavy deps required -- vehicle_registry only uses the stdlib.)
"""
from app.detection.vehicle_registry import VehicleRegistry, iou


def registry(**kw) -> VehicleRegistry:
    opts = dict(
        reid_gap=4.0, stationary_seconds=20.0, forget_seconds=60.0,
        parked_memory=300.0,
    )
    opts.update(kw)
    return VehicleRegistry(**opts)


def park(r, box=None, track_id=1, start=0.0, seconds=40.0, step=1.0):
    """Hold a vehicle still long enough to be judged parked; return it."""
    box = box if box is not None else PARKED
    t = start
    v = None
    while t <= start + seconds:
        v = r.observe(track_id, box, now=t)
        t += step
    return v


def shifted(box, dx=0.0, dy=0.0):
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


PARKED = (200.0, 400.0, 520.0, 740.0)  # roughly the car in the reported frame


def test_renumbered_vehicle_keeps_its_state():
    """A new track id on the same box is the same vehicle, not a new one."""
    r = registry()
    v1 = r.observe(1, PARKED, now=0.0)
    v1.attempts = 12
    v1.best_conf = 0.29
    v1.text = "G2JI"
    v1.row_id = 77
    # Tracker loses it and picks it up again a second later as id 2.
    v2 = r.observe(2, shifted(PARKED, dx=2.0), now=1.0)
    assert v2.vid == v1.vid, (v1.vid, v2.vid)
    assert v2.attempts == 12 and v2.row_id == 77, v2
    print("OK renumbered_vehicle_keeps_its_state: id 1 -> 2 kept row 77")


def test_new_vehicle_after_the_gap_is_new():
    """The spot freeing up and refilling later is the next car, not the same one."""
    r = registry(reid_gap=4.0)
    first = r.observe(1, PARKED, now=0.0)
    second = r.observe(2, PARKED, now=30.0)
    assert second.vid != first.vid, (first.vid, second.vid)
    assert second.row_id is None and second.attempts == 0, second
    print("OK new_vehicle_after_the_gap_is_new: 30s later is a different vehicle")


def test_two_vehicles_in_one_frame_stay_apart():
    """A vehicle already seen this frame cannot be adopted by another track."""
    r = registry()
    a = r.observe(1, PARKED, now=0.0)
    b = r.observe(2, shifted(PARKED, dx=330.0), now=0.0)
    assert a.vid != b.vid, (a.vid, b.vid)
    # Same frame, and the id that would otherwise match `a` is already taken.
    c = r.observe(3, PARKED, now=0.0)
    assert c.vid != a.vid, (a.vid, c.vid)
    print("OK two_vehicles_in_one_frame_stay_apart")


def test_stopped_vehicle_is_read_then_left_alone():
    """The reported bug, end to end: arrive, be read, then stop being read."""
    r = registry(stationary_seconds=20.0)
    t = 0.0
    v = r.observe(1, PARKED, now=t)
    assert not r.is_parked(v, t), "must be readable when it first shows up"
    # Sits still, and the tracker renumbers it every few seconds as it flickers.
    reads_allowed = 0
    for i in range(1, 60):
        t = i * 2.0
        v = r.observe(1 + i // 3, shifted(PARKED, dx=(i % 2) * 3.0), now=t)
        if not r.is_parked(v, t):
            reads_allowed += 1
    # Only the first stationary_seconds worth of frames may be read; after that
    # the car is parked and nothing more is written, however often it is
    # renumbered. Before the fix every renumbering was a new row.
    assert 0 < reads_allowed <= 10, reads_allowed
    assert r.is_parked(v, t), "still being read after two minutes parked"
    print(f"OK stopped_vehicle_is_read_then_left_alone: {reads_allowed} readable frames, then quiet")


def test_creeping_vehicle_is_not_parked():
    """A car edging forward in a queue is moving, however slowly."""
    r = registry(stationary_seconds=20.0)
    box = PARKED
    t = 0.0
    # 3 px per second on a 320 px-wide box: under the 15% threshold between any
    # two consecutive frames, so a frame-to-frame test would call this parked.
    for i in range(1, 121):
        t = i * 1.0
        box = shifted(box, dy=3.0)
        v = r.observe(1, box, now=t)
    assert not r.is_parked(v, t), "creeping vehicle judged parked"
    print("OK creeping_vehicle_is_not_parked: 3 px/s stayed readable for 2 min")


def test_vehicle_approaching_the_camera_is_not_parked():
    """Driving at the lens grows the box without moving its centre much."""
    r = registry(stationary_seconds=20.0)
    t = 0.0
    for i in range(1, 61):
        t = i * 1.0
        grow = i * 4.0
        v = r.observe(1, (300.0 - grow, 400.0 - grow, 620.0 + grow, 740.0 + grow), now=t)
    assert not r.is_parked(v, t), "approaching vehicle judged parked"
    print("OK vehicle_approaching_the_camera_is_not_parked")


def test_jitter_does_not_count_as_movement():
    """Detector noise on a stationary car must not keep it readable forever."""
    r = registry(stationary_seconds=20.0)
    t = 0.0
    for i in range(1, 121):
        t = i * 1.0
        # +-6 px on a 320 px box: 2%, well inside real detector jitter.
        wobble = 6.0 if i % 2 else -6.0
        v = r.observe(1, shifted(PARKED, dx=wobble, dy=-wobble), now=t)
    assert r.is_parked(v, t), "jitter alone kept the vehicle readable"
    print("OK jitter_does_not_count_as_movement")


def test_parked_vehicle_becomes_readable_again_when_it_leaves():
    r = registry(stationary_seconds=20.0)
    t = 0.0
    for i in range(1, 41):
        t = i * 1.0
        v = r.observe(1, PARKED, now=t)
    assert r.is_parked(v, t)
    v = r.observe(1, shifted(PARKED, dy=-120.0), now=t + 1.0)
    assert not r.is_parked(v, t + 1.0), "pulling away did not resume reading"
    print("OK parked_vehicle_becomes_readable_again_when_it_leaves")


def test_parked_car_the_detector_lost_comes_back_as_itself():
    """The reported gaps: reads 2-4 minutes apart, i.e. the car was not merely
    renumbered, it had dropped out of detection entirely in between."""
    r = registry()
    v = park(r, start=0.0, seconds=40.0)
    v.attempts, v.best_conf, v.text, v.row_id = 12, 0.29, "G2JI", 41
    assert r.is_parked(v, 40.0)
    # Gone from every frame for four minutes, then detected again in place.
    again = r.observe(99, shifted(PARKED, dx=3.0), now=280.0)
    assert again.vid == v.vid, "reappearing parked car counted as a new vehicle"
    assert again.row_id == 41 and again.attempts == 12, again
    assert r.is_parked(again, 280.0), "should stay quiet, not be read afresh"
    print("OK parked_car_the_detector_lost_comes_back_as_itself: 4 min gap, same row")


def test_parked_memory_expires():
    """Long enough and the spot is simply available to whoever is in it now."""
    r = registry(parked_memory=300.0)
    v = park(r, start=0.0, seconds=40.0)
    later = r.observe(99, PARKED, now=40.0 + 400.0)
    assert later.vid != v.vid, "a parked car is remembered forever"
    print("OK parked_memory_expires: past the window it is a new vehicle")


def test_long_gap_demands_a_near_exact_box():
    """A merely similar box after a long absence is a different vehicle."""
    r = registry()
    v = park(r, start=0.0, seconds=40.0)
    # ~30% of the box away: plausible as the next car in the same spot, and not
    # good enough to inherit a parked car's identity minutes later.
    other = r.observe(99, shifted(PARKED, dx=100.0), now=200.0)
    assert other.vid != v.vid, "loose match adopted across a long gap"
    print("OK long_gap_demands_a_near_exact_box")


def test_car_pausing_at_a_gate_is_not_given_parked_memory():
    """Short stops must not inherit the long window, or a queue merges into one."""
    r = registry(stationary_seconds=20.0)
    first = park(r, track_id=1, start=0.0, seconds=10.0)  # stopped 10s: not parked
    assert not r.is_parked(first, 10.0)
    # It drives off, and half a minute later the next car stops in the same spot.
    second = r.observe(2, PARKED, now=45.0)
    assert second.vid != first.vid, "two cars at a gate merged into one row"
    print("OK car_pausing_at_a_gate_is_not_given_parked_memory")


def test_forgotten_vehicles_are_dropped():
    """The registry must not grow for the lifetime of the worker."""
    r = registry(forget_seconds=60.0)
    for i in range(200):
        r.observe(i, shifted(PARKED, dx=i * 400.0), now=float(i))
    r.prune(now=250.0)
    assert len(r._vehicles) <= 61, len(r._vehicles)
    assert all(tid in r._by_track for tid in (190, 199)), "live tracks dropped"
    print(f"OK forgotten_vehicles_are_dropped: {len(r._vehicles)} of 200 kept")


def test_labels_follow_the_vehicle():
    r = registry()
    v = r.observe(1, PARKED, now=0.0)
    v.text = "B2356UOZ"
    assert r.labels() == {1: "B2356UOZ"}
    r.observe(2, PARKED, now=1.0)
    assert r.labels() == {2: "B2356UOZ"}, r.labels()
    print("OK labels_follow_the_vehicle: overlay survives renumbering")


def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    print("OK iou_basics")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} vehicle-registry tests passed.")
