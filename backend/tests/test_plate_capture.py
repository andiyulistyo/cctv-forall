"""Tests for capturing a vehicle that passes through the ANPR zone unread.

What this pins down: a motorcycle plate is usually unreadable on an overview
camera, so ALPR_CLASSES leaves motorcycles out entirely -- and "we could not
read its plate" ends up recorded as "nothing came past". ALPR_CAPTURE_CLASSES
records the passage itself: the vehicle crop and the frame go in, plate_text
stays empty, and a plate fills it in later if one can be read.

The two properties worth defending are the ones that are easy to break later:

* a capture happens **only inside the zone** -- with no zone drawn, not at all;
* a capture **never takes a plate read's place** in the queue, so the classes
  in ALPR_CLASSES keep exactly the OCR budget they had before.

Run from the backend/ directory:  python -m tests.test_plate_capture
"""
import atexit
import queue
import threading
import time

import numpy as np

import app.detection.worker as worker
from app.detection.detector import Detection
from app.detection.worker import _PlateReader, _capture_classes


class Clock:
    """A monotonic clock the test drives, so frames are frames.

    ``submit()`` reads ``time.monotonic()`` itself, and on Windows that has a
    ~16 ms granularity -- two submits in a loop routinely return the *same*
    reading, which the registry rightly treats as two boxes in one frame rather
    than one vehicle across two. Without this, a test about renumbering would
    be testing the timer.
    """

    def __init__(self, start: float = 1000.0):
        self.t = start

    def monotonic(self) -> float:
        return self.t

    def tick(self, dt: float = 0.1) -> None:
        self.t += dt

    def __getattr__(self, name):  # anything else stays the real time module
        return getattr(time, name)


# reader() swaps this clock into the worker module. Put the real one back when
# the process ends, so importing this file cannot leave a frozen monotonic()
# behind for whatever runs next.
atexit.register(lambda: setattr(worker, "time", time))


FRAME_H, FRAME_W = 720, 1280
# Bottom half of the frame, the way ZoneDrawCanvas stores it.
BOTTOM_HALF = {"a": [0.0, 0.5], "b": [1.0, 1.0]}


class StubALPR:
    """Stands in for the real ALPR: counts reads, returns whatever it is given."""

    def __init__(self, result=None):
        self.result = result
        self.calls = 0

    def read_plate(self, crop):
        self.calls += 1
        return self.result


def reader(**kw) -> _PlateReader:
    """A reader wired to a stub ALPR, a test clock, and a recording _offer."""
    worker.time = Clock()
    opts = dict(
        alpr=StubALPR(),
        source_id=1,
        classes=("car", "truck", "bus"),
        min_vehicle_width=160,
        capture_classes=("motorcycle",),
        capture_min_width=48,
        zone=BOTTOM_HALF,
        save_frame=False,
    )
    opts.update(kw)
    r = _PlateReader(**opts)
    # Intercept the hand-off so nothing reaches the OCR thread or the database:
    # every test here is about which jobs are offered, and on what terms.
    r.offered = []

    def record(vid, job, evict=True):
        r.offered.append((job[0], vid, evict))
        return True          # _offer reports whether the queue took the job

    r._offer = record
    r.clock = worker.time
    return r


def next_frame(r) -> None:
    """Advance to the next frame, as the detection loop would."""
    r.clock.tick()


def frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def bike(track_id=1, x1=600, y1=500, x2=680, y2=600):
    """A motorcycle-sized box, low in the frame -- inside the bottom-half zone."""
    return Detection(track_id, "motorcycle", 0.9, x1, y1, x2, y2)


def car(track_id=2, x1=300, y1=400, x2=700, y2=650):
    """A car wide enough to clear ALPR_MIN_VEHICLE_WIDTH."""
    return Detection(track_id, "car", 0.9, x1, y1, x2, y2)


def test_motorcycle_in_the_zone_is_captured():
    r = reader()
    r.submit(frame(), [bike()])
    assert r.offered == [("capture", r.offered[0][1], False)], r.offered
    print("OK motorcycle_in_the_zone_is_captured")


def test_motorcycle_outside_the_zone_is_never_captured():
    """The whole point of the feature: in the zone, or not at all."""
    r = reader()
    # Same bike, high in the frame -- above the bottom-half zone.
    r.submit(frame(), [bike(y1=100, y2=200)])
    assert r.offered == [], r.offered
    print("OK motorcycle_outside_the_zone_is_never_captured")


def test_no_zone_means_no_capture_at_all():
    """No zone drawn is not "the whole frame is the zone"."""
    r = reader(zone=None)
    r.submit(frame(), [bike()])
    assert r.offered == [], r.offered
    print("OK no_zone_means_no_capture_at_all")


def test_a_vehicle_is_captured_once_not_once_per_frame():
    r = reader()
    f = frame()
    for _ in range(10):
        r.submit(f, [bike()])
        next_frame(r)
    kinds = [o[0] for o in r.offered]
    assert kinds == ["capture"], kinds
    print(f"OK a_vehicle_is_captured_once_not_once_per_frame: {len(kinds)} of 10 frames")


def test_renumbered_motorcycle_is_not_captured_twice():
    """The tracker renumbering a bike must not read as a second bike."""
    r = reader()
    f = frame()
    r.submit(f, [bike(track_id=1)])
    next_frame(r)
    r.submit(f, [bike(track_id=2, x1=602, x2=682)])  # same box, new id
    kinds = [o[0] for o in r.offered]
    assert kinds == ["capture"], kinds
    print("OK renumbered_motorcycle_is_not_captured_twice: id 1 -> 2 captured once")


def test_a_capture_never_evicts_a_plate_read():
    """The guarantee that ALPR_CLASSES keeps the budget it had before."""
    r = reader()
    r.submit(frame(), [bike(), car()])
    by_kind = {kind: evict for kind, _vid, evict in r.offered}
    assert by_kind.get("capture") is False, r.offered
    assert by_kind.get("read") is True, r.offered
    print("OK a_capture_never_evicts_a_plate_read")


def test_too_small_to_be_worth_capturing_is_skipped():
    r = reader(capture_min_width=48)
    r.submit(frame(), [bike(x1=600, x2=630)])  # 30 px wide
    assert r.offered == [], r.offered
    print("OK too_small_to_be_worth_capturing_is_skipped")


def test_capture_uses_its_own_width_floor_not_the_read_one():
    """160 px is about plate legibility and would reject every motorcycle."""
    r = reader()
    r.submit(frame(), [bike(x1=600, x2=680)])  # 80 px: under 160, over 48
    assert [o[0] for o in r.offered] == ["capture"], r.offered
    print("OK capture_uses_its_own_width_floor_not_the_read_one")


def test_classes_not_configured_for_capture_are_untouched():
    """With the setting empty, this is the behaviour that shipped before."""
    r = reader(capture_classes=())
    r.submit(frame(), [bike(), car()])
    assert [o[0] for o in r.offered] == ["read"], r.offered
    print("OK classes_not_configured_for_capture_are_untouched")


def test_a_read_class_is_read_not_captured():
    """Listed in both settings, a class keeps its full attempt budget."""
    r = reader(classes=("car", "motorcycle"), capture_classes=("motorcycle",),
               min_vehicle_width=0)
    r.submit(frame(), [bike()])
    assert [o[0] for o in r.offered] == ["read"], r.offered
    print("OK a_read_class_is_read_not_captured")


class BlockingALPR:
    """Holds the ANPR thread on its first job so the queue can be kept full."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def read_plate(self, crop):
        self.entered.set()
        self.release.wait(timeout=10)
        return None


def test_a_dropped_capture_is_retried_not_lost():
    """A full queue means "busy this frame", not "this bike was never here"."""
    alpr = BlockingALPR()
    r = reader(alpr=alpr)
    del r._offer  # use the real one: the queue rules are what is under test
    try:
        # Park the worker thread inside a read, then fill what is left. The
        # thread pulls one job the moment it is queued, so "maxsize puts" is
        # not the same as "queue full" -- push until it says so itself.
        r._queue.put_nowait(("read", 900, 900, "car", None, None, None))
        assert alpr.entered.wait(timeout=5), "worker thread never started its job"
        n = 0
        while True:
            try:
                r._queue.put_nowait(("read", 910 + n, 910 + n, "car", None, None, None))
                n += 1
            except queue.Full:
                break

        r.submit(frame(), [bike()])
        assert r._queue.full(), "a capture must not have displaced a read"
        vehicle = next(iter(r._vehicles._vehicles.values()))
        assert vehicle.captured is False, "a dropped capture must not count as done"

        # A slot frees up; the next frame gets the capture in.
        r._queue.get_nowait()
        next_frame(r)
        r.submit(frame(), [bike()])
        assert vehicle.captured is True, "the retry should have been queued"
        kinds = [r._queue.get_nowait()[0] for _ in range(r._queue.qsize())]
        assert "capture" in kinds, kinds
        print("OK a_dropped_capture_is_retried_not_lost")
    finally:
        alpr.release.set()


# The taxi that was recorded four times in seven seconds on camera 20, as
# boxes measured off the saved evidence frames (3840x2160) against the zone the
# operator had drawn. Only the first is meaningfully inside it.
ALLEY_ZONE = {"a": [0.10983605382520062, 0.5093687241742849],
              "b": [0.7147540563487541, 0.9918136834443947]}
TAXI_BOXES = {                       # track id -> box, and how much is in zone
    215: ((390, 1182, 1539, 2151), 0.96),
    216: ((1018, 763, 1701, 1366), 0.44),
    220: ((1149, 667, 1780, 1198), 0.19),
    223: ((1258, 639, 1963, 1122), 0.05),
}


def test_zone_overlap_matches_the_measured_frames():
    """The arithmetic, checked against boxes taken off the real evidence frames."""
    zone = worker._zone_px(ALLEY_ZONE, 3840, 2160)
    for track, (box, expected) in TAXI_BOXES.items():
        got = worker._zone_overlap(box, zone)
        assert abs(got - expected) < 0.02, (track, got, expected)
    print("OK zone_overlap_matches_the_measured_frames")


def test_only_the_vehicle_actually_in_the_zone_is_read():
    """The bug from camera 20: one car, four rows, three of them barely in zone."""
    r = reader(zone=ALLEY_ZONE)
    f = np.zeros((2160, 3840, 3), dtype=np.uint8)
    verdicts = {t: r._in_zone(f, box) for t, (box, _) in TAXI_BOXES.items()}
    assert verdicts == {215: True, 216: False, 220: False, 223: False}, verdicts
    print("OK only_the_vehicle_actually_in_the_zone_is_read: 1 of 4 frames")


def test_a_car_whose_wheels_pass_the_lower_edge_is_still_in_the_zone():
    """The old point test threw this one away -- it is 96% inside."""
    r = reader(zone=ALLEY_ZONE)
    f = np.zeros((2160, 3840, 3), dtype=np.uint8)
    box, _ = TAXI_BOXES[215]
    ground_y = box[3]
    assert ground_y > ALLEY_ZONE["b"][1] * 2160, "the ground point is below the zone"
    assert r._in_zone(f, box), "but the car is squarely inside it"
    print("OK a_car_whose_wheels_pass_the_lower_edge_is_still_in_the_zone")


def test_zone_min_overlap_is_configurable():
    f = np.zeros((2160, 3840, 3), dtype=np.uint8)
    box, _ = TAXI_BOXES[220]                     # 19% inside
    assert not reader(zone=ALLEY_ZONE, zone_min_overlap=0.5)._in_zone(f, box)
    assert reader(zone=ALLEY_ZONE, zone_min_overlap=0.1)._in_zone(f, box)
    print("OK zone_min_overlap_is_configurable")


def test_capture_classes_parsing():
    assert _capture_classes("motorcycle") == ("motorcycle",)
    assert _capture_classes(" Motorcycle , bus ") == ("motorcycle", "bus")
    # Empty means none, not "everything" -- the opposite of ALPR_CLASSES.
    assert _capture_classes("") == ()
    assert _capture_classes("bicycle") == ()
    print("OK capture_classes_parsing")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
