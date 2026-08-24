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

    r.captured_match = None

    def record(vid, job, evict=True):
        r.offered.append((job[0], vid, evict))
        if job[0] == "capture":
            r.captured_match = job[7]
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


# The motorcycle recorded twice on camera 22, as boxes measured off its two
# saved evidence frames (3840x2160) against the zone that camera's operator had
# drawn. Both rows carry the plate B 6984 NK, 1.19 s apart, under tracker ids
# 155 and 157. Their boxes overlap by IoU 0.17, far under the registry's
# re-identification threshold, so nothing about identity could have joined them:
# by the time the tracker had renumbered the bike it had moved most of its own
# length. What separates them is how far into the zone each one is.
GATE_ZONE = {"a": [0.2512294893744052, 0.5879570433314069],
             "b": [0.7786884793796988, 0.9983627100408214]}
BIKE_ENTERING = (1492, 1952, 1769, 2159)   # 98.6% in zone: clipped by the frame
BIKE_THROUGH = (1492, 1702, 1725, 2037)    # 100%: the whole machine, in the zone


def test_a_bike_straddling_the_zone_edge_is_not_captured_yet():
    """The first of the two rows: recorded on the way out, cut off by the frame."""
    r = reader(zone=GATE_ZONE)
    f = np.zeros((2160, 3840, 3), dtype=np.uint8)
    r.submit(f, [bike(track_id=155, x1=BIKE_ENTERING[0], y1=BIKE_ENTERING[1],
                      x2=BIKE_ENTERING[2], y2=BIKE_ENTERING[3])])
    assert r.offered == [], r.offered
    print("OK a_bike_straddling_the_zone_edge_is_not_captured_yet")


def test_the_bike_is_captured_once_it_is_wholly_through():
    """...and the second row, which is the one worth keeping: the whole bike."""
    r = reader(zone=GATE_ZONE)
    f = np.zeros((2160, 3840, 3), dtype=np.uint8)
    r.submit(f, [bike(track_id=155, x1=BIKE_ENTERING[0], y1=BIKE_ENTERING[1],
                      x2=BIKE_ENTERING[2], y2=BIKE_ENTERING[3])])
    next_frame(r)
    # Renumbered to 157, as the tracker actually did, and now wholly in the zone.
    r.submit(f, [bike(track_id=157, x1=BIKE_THROUGH[0], y1=BIKE_THROUGH[1],
                      x2=BIKE_THROUGH[2], y2=BIKE_THROUGH[3])])
    assert [o[0] for o in r.offered] == ["capture"], r.offered
    print("OK the_bike_is_captured_once_it_is_wholly_through: 1 row, not 2")


# The other way one bike became two rows, on camera 20 this time: the detector
# returned the rider and machine together as one box (track 7) and the machine
# alone as another (track 8), 43 ms apart, both wholly inside the zone. They
# score IoU 0.37 -- under the registry's threshold -- so the zone rule cannot
# help here and the ledger has to.
RIDER_AND_BIKE = (1294, 1191, 1601, 1787)
BIKE_ALONE = (1378, 1294, 1571, 1639)


def test_one_bike_reported_as_two_nested_boxes_is_captured_once():
    r = reader(zone=ALLEY_ZONE)
    f = np.zeros((2160, 3840, 3), dtype=np.uint8)
    r.submit(f, [
        bike(track_id=7, x1=RIDER_AND_BIKE[0], y1=RIDER_AND_BIKE[1],
             x2=RIDER_AND_BIKE[2], y2=RIDER_AND_BIKE[3]),
        bike(track_id=8, x1=BIKE_ALONE[0], y1=BIKE_ALONE[1],
             x2=BIKE_ALONE[2], y2=BIKE_ALONE[3]),
    ])
    assert [o[0] for o in r.offered] == ["capture"], r.offered
    print("OK one_bike_reported_as_two_nested_boxes_is_captured_once")


def test_two_bikes_side_by_side_are_both_captured():
    """The ledger must suppress a repeat, not the second half of the traffic."""
    r = reader(zone=ALLEY_ZONE)
    f = np.zeros((2160, 3840, 3), dtype=np.uint8)
    r.submit(f, [
        bike(track_id=11, x1=1294, y1=1191, x2=1601, y2=1787),
        bike(track_id=12, x1=1900, y1=1191, x2=2207, y2=1787),  # alongside it
    ])
    assert [o[0] for o in r.offered] == ["capture", "capture"], r.offered
    print("OK two_bikes_side_by_side_are_both_captured")


def test_the_gate_can_be_widened_back_to_a_region():
    """Operators with a tight zone need the old behaviour available."""
    f = np.zeros((2160, 3840, 3), dtype=np.uint8)
    entering = bike(track_id=155, x1=BIKE_ENTERING[0], y1=BIKE_ENTERING[1],
                    x2=BIKE_ENTERING[2], y2=BIKE_ENTERING[3])
    r = reader(zone=GATE_ZONE, capture_containment=0.5)
    r.submit(f, [entering])
    assert [o[0] for o in r.offered] == ["capture"], r.offered
    print("OK the_gate_can_be_widened_back_to_a_region")


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
        r._queue.put_nowait(("read", 900, 900, "car", None, None, None, None))
        assert alpr.entered.wait(timeout=5), "worker thread never started its job"
        n = 0
        while True:
            try:
                r._queue.put_nowait(("read", 910 + n, 910 + n, "car", None, None, None, None))
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


# ---------------------------------------------------------------- person
def person(track_id=3, x1=600, y1=440, x2=700, y2=620):
    """A person-shaped box standing in the bottom-half zone."""
    return Detection(track_id, "person", 0.9, x1, y1, x2, y2)


def face(x=630, y=450, w=30, h=30, name="BUDI", sim=0.88):
    """One entry as _run_faces returns it: (x, y, w, h, name, similarity)."""
    return (x, y, w, h, name, sim)


def person_reader(**kw):
    r = reader(capture_classes=("motorcycle", "person"), **kw)
    r.sunk = []
    r._person_sink = lambda name, sim, crop, snap, box: r.sunk.append((name, sim))
    return r


def test_person_in_the_zone_is_captured():
    r = person_reader()
    r.submit(frame(), [person()])
    assert [o[0] for o in r.offered] == ["capture"], r.offered
    print("OK person_in_the_zone_is_captured")


def test_person_outside_the_zone_is_never_captured():
    """Same rule as every other capture: in the zone, or not at all."""
    r = person_reader()
    r.submit(frame(), [person(y1=60, y2=240)])   # above the bottom-half zone
    assert r.offered == [], r.offered
    print("OK person_outside_the_zone_is_never_captured")


def test_person_capture_carries_the_recognized_name():
    """The face the frame was already scanned for gets attached to the capture."""
    r = person_reader()
    r.submit(frame(), [person()], faces=[face()])
    job_match = r.captured_match
    assert job_match == ("BUDI", 0.88), job_match
    print("OK person_capture_carries_the_recognized_name")


def test_person_capture_with_no_face_is_still_captured():
    """Nobody turned to the camera is still somebody who came through."""
    r = person_reader()
    r.submit(frame(), [person()], faces=[])
    assert [o[0] for o in r.offered] == ["capture"], r.offered
    assert r.captured_match is None, r.captured_match
    print("OK person_capture_with_no_face_is_still_captured")


def test_a_face_outside_the_person_box_is_not_theirs():
    """Two people in frame must not swap identities."""
    r = person_reader()
    r.submit(frame(), [person()], faces=[face(x=100, y=450)])
    assert r.captured_match is None, r.captured_match
    print("OK a_face_outside_the_person_box_is_not_theirs")


def test_a_named_face_beats_an_unnamed_one():
    r = person_reader()
    r.submit(frame(), [person()],
             faces=[face(x=610, y=450, name=None, sim=0.99), face(x=660, y=450)])
    assert r.captured_match == ("BUDI", 0.88), r.captured_match
    print("OK a_named_face_beats_an_unnamed_one")


def test_faces_are_scaled_into_the_full_frame():
    """Faces arrive in detection-frame coords; boxes are in full-frame coords."""
    r = person_reader()
    # scale 2.0: the detection frame is half the size of the frame handed in.
    # Detection frame is 1280x720, the real frame 2560x1440; the person stands
    # in the lower half of both, which is where the zone is.
    r.submit(np.zeros((1440, 2560, 3), dtype=np.uint8),
             [Detection(3, "person", 0.9, 300, 400, 350, 560)],
             scale=2.0, faces=[face(x=315, y=405, w=15, h=15)])
    assert r.captured_match == ("BUDI", 0.88), r.captured_match
    print("OK faces_are_scaled_into_the_full_frame")


def test_person_goes_to_the_sink_not_the_plate_table():
    """A person is a face sighting; only vehicles become plate rows."""
    alpr = StubALPR()
    r = person_reader(alpr=alpr)
    r._do_capture(1, 3, "person", np.zeros((10, 10, 3), np.uint8), None,
                  (0, 0, 10, 10), ("BUDI", 0.88))
    assert r.sunk == [("BUDI", 0.88)], r.sunk
    print("OK person_goes_to_the_sink_not_the_plate_table")


# ------------------------------------------------- the sink and its throttle
class FakeFaceState:
    """_FaceState's logging throttle, without the models behind it."""

    def __init__(self, cooldown=20, log_unknown=False):
        self.cooldown = cooldown
        self.log_unknown = log_unknown
        self.last_logged = {}
        self._log_lock = threading.Lock()

    claim_log = worker._FaceState.claim_log


def sink_with(monkey_rows, fstate, evidence=None):
    """_person_sink, with _persist_sighting captured instead of written.

    ``evidence`` collects the full-frame arguments, which most of these tests
    do not care about but one of them is entirely about.
    """
    real = worker._persist_sighting

    def capture(src, name, sim, crop, snapshot=None, box=None):
        monkey_rows.append((name, sim))
        if evidence is not None:
            evidence.append((snapshot, box))

    worker._persist_sighting = capture
    try:
        return worker._person_sink(1, fstate), real
    except Exception:
        worker._persist_sighting = real
        raise


def test_a_recognized_person_is_not_filed_twice():
    """The frame-wide face pass logged BUDI; the zone capture must stand down."""
    rows = []
    fs = FakeFaceState()
    sink, real = sink_with(rows, fs)
    try:
        assert fs.claim_log("BUDI", time.time())      # the face pass got there first
        sink("BUDI", 0.88, None, None, None)
        assert rows == [], rows
    finally:
        worker._persist_sighting = real
    print("OK a_recognized_person_is_not_filed_twice")


def test_a_recognized_person_is_filed_when_the_face_pass_missed_them():
    rows = []
    fs = FakeFaceState()
    sink, real = sink_with(rows, fs)
    try:
        sink("BUDI", 0.88, None, None, None)
        assert rows == [("BUDI", 0.88)], rows
    finally:
        worker._persist_sighting = real
    print("OK a_recognized_person_is_filed_when_the_face_pass_missed_them")


def test_a_person_capture_carries_its_evidence_frame():
    """The snapshot the capture already copied reaches the sighting row.

    A face crop says who was seen. Only the frame says where they were and who
    was with them, which is the whole reason the Wajah page can show one -- and
    the capture path is the one that already holds a snapshot, so nothing extra
    is copied to get it there.
    """
    rows, evidence = [], []
    fs = FakeFaceState()
    sink, real = sink_with(rows, fs, evidence)
    try:
        snapshot = np.zeros((48, 64, 3), dtype=np.uint8)
        sink("BUDI", 0.88, None, snapshot, (4, 6, 20, 30))
        assert evidence == [(snapshot, (4, 6, 20, 30))], evidence
    finally:
        worker._persist_sighting = real
    print("OK a_person_capture_carries_its_evidence_frame")


def test_an_unrecognized_person_is_always_filed():
    """The capability the frame-wide pass cannot provide: a face it never saw."""
    rows = []
    fs = FakeFaceState(log_unknown=False)
    sink, real = sink_with(rows, fs)
    try:
        sink(None, 0.0, None, None, None)
        sink(None, 0.0, None, None, None)   # a second, different person
        assert rows == [(None, 0.0), (None, 0.0)], rows
    finally:
        worker._persist_sighting = real
    print("OK an_unrecognized_person_is_always_filed")


def test_unknowns_respect_the_throttle_when_the_face_pass_logs_them_too():
    """With FACE_LOG_UNKNOWN on there *is* something to collide with."""
    rows = []
    fs = FakeFaceState(log_unknown=True)
    sink, real = sink_with(rows, fs)
    try:
        sink(None, 0.0, None, None, None)
        sink(None, 0.0, None, None, None)
        assert rows == [(None, 0.0)], rows
    finally:
        worker._persist_sighting = real
    print("OK unknowns_respect_the_throttle_when_the_face_pass_logs_them_too")


def test_person_capture_works_with_face_recognition_off():
    """No recognizer on this source: still record that someone came through."""
    rows = []
    sink, real = sink_with(rows, None)
    try:
        sink(None, 0.0, None, None, None)
        assert rows == [(None, 0.0)], rows
    finally:
        worker._persist_sighting = real
    print("OK person_capture_works_with_face_recognition_off")


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
