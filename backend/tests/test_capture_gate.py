"""Tests for the rule that decides a capture is a new passage, not a repeat.

The vehicle registry stops a capture repeating for as long as it recognises the
vehicle. On a zoomed-in overview camera it often cannot, and the two ways it
fails were both measured on this project's own recordings -- see
``app/detection/capture_gate.py`` for the numbers. This file pins down the two
rules that cover them, using the real geometry from those recordings:

* a vehicle straddling the edge of the zone has not been through it yet;
* a vehicle reported as two nested boxes is one vehicle, not two.

And, just as importantly, that neither rule swallows traffic it should record.

Run from the backend/ directory:  python -m tests.test_capture_gate
"""
from app.detection.capture_gate import CaptureGate, containment

# Both cameras in the sample recordings are 4K; the zones below are the ones
# their operator actually drew, converted to pixels.
ZONE_7 = (965.0, 1270.0, 2990.0, 2156.0)   # "Kamera 22", alpr_zone on source 7
ZONE_6 = (422.0, 1100.0, 2745.0, 2142.0)   # "Kamera 20", alpr_zone on source 6


def gate(**kw) -> CaptureGate:
    opts = dict(min_containment=0.99, window=3.0, min_overlap=0.8)
    opts.update(kw)
    return CaptureGate(**opts)


# --- containment -----------------------------------------------------------

def test_containment_is_the_fraction_of_the_vehicle_inside_the_zone():
    box = (1000.0, 1300.0, 1200.0, 1500.0)  # wholly inside ZONE_7
    assert containment(box, ZONE_7) == 1.0
    half = (865.0, 1300.0, 1065.0, 1500.0)  # 100 px of 200 past the left edge
    assert abs(containment(half, ZONE_7) - 0.5) < 1e-9
    print("OK containment_is_the_fraction_of_the_vehicle_inside_the_zone")


def test_containment_is_zero_when_they_do_not_meet():
    assert containment((0.0, 0.0, 100.0, 100.0), ZONE_7) == 0.0
    print("OK containment_is_zero_when_they_do_not_meet")


def test_a_vehicle_bigger_than_the_zone_can_still_fill_it():
    """Otherwise a tight zone, or a lorry up close, could never be captured."""
    huge = (0.0, 0.0, 3840.0, 2160.0)  # the whole frame, swallowing the zone
    assert containment(huge, ZONE_7) == 1.0
    assert gate().is_through(huge, ZONE_7)
    print("OK a_vehicle_bigger_than_the_zone_can_still_fill_it")


# --- the zone as a gate ----------------------------------------------------

def test_a_bike_clipped_by_the_frame_edge_has_not_been_through():
    """Row 432: the first of two rows written for plate B 6984 NK.

    98.77% inside the zone -- a motorcycle on its way out of shot, cut off by
    the bottom of the frame. Under the old half-in rule this was recorded, and
    then recorded again 1.19 s later under a new tracker id.
    """
    clipped = (1492.0, 1952.0, 1769.0, 2159.0)
    assert abs(containment(clipped, ZONE_7) - 0.98551) < 1e-4
    assert not gate().is_through(clipped, ZONE_7)
    print("OK a_bike_clipped_by_the_frame_edge_has_not_been_through")


def test_the_other_measured_duplicate_is_rejected_too():
    """Row 429, the first of the pair 1.26 s apart, whose boxes did not touch.

    Its partner scored IoU 0.00 against it: the bike had moved clean past its
    own box between the two detections, which is what put re-identification out
    of reach and made the zone rule the one that had to catch it.
    """
    clipped = (1208.0, 1946.0, 1571.0, 2159.0)
    assert abs(containment(clipped, ZONE_7) - 0.98592) < 1e-4
    assert not gate().is_through(clipped, ZONE_7)
    print("OK the_other_measured_duplicate_is_rejected_too")


def test_the_same_bike_a_second_later_is_through():
    """Row 433: the duplicate. Wholly inside, and the better evidence of the two."""
    inside = (1492.0, 1702.0, 1725.0, 2037.0)
    assert containment(inside, ZONE_7) == 1.0
    assert gate().is_through(inside, ZONE_7)
    print("OK the_same_bike_a_second_later_is_through")


def test_the_gate_leaves_room_for_detector_jitter():
    """A box a pixel over the edge is inside by any reading that matters."""
    jittered = (1492.0, 1702.0, 1725.0, 2157.0)  # 1 px past ZONE_7's lower edge
    assert containment(jittered, ZONE_7) < 1.0
    assert gate().is_through(jittered, ZONE_7)
    assert not gate(min_containment=1.0).is_through(jittered, ZONE_7)
    print("OK the_gate_leaves_room_for_detector_jitter")


def test_a_half_in_vehicle_is_never_through():
    half_in = (1492.0, 2000.0, 1769.0, 2400.0)  # well past the bottom edge
    assert not gate().is_through(half_in, ZONE_7)
    print("OK a_half_in_vehicle_is_never_through")


# --- the ledger ------------------------------------------------------------

def test_a_nested_detection_is_the_same_vehicle():
    """Rows 425 and 426: plate B 6285, 43 ms apart, both wholly inside ZONE_6.

    One box around the rider and machine together, one around the machine
    alone. They score IoU 0.37 -- under the registry's re-identification
    threshold -- while the smaller sits entirely inside the larger.
    """
    rider_and_bike = (1294.0, 1191.0, 1601.0, 1787.0)
    bike_alone = (1378.0, 1294.0, 1571.0, 1639.0)
    assert containment(bike_alone, rider_and_bike) == 1.0

    g = gate()
    assert not g.is_repeat("motorcycle", rider_and_bike, 100.0)
    g.record("motorcycle", rider_and_bike, 100.0)
    assert g.is_repeat("motorcycle", bike_alone, 100.04)
    print("OK a_nested_detection_is_the_same_vehicle")


def test_a_different_vehicle_elsewhere_in_the_zone_is_still_captured():
    """The rule must not turn into "one capture per zone per few seconds"."""
    g = gate()
    g.record("motorcycle", (1294.0, 1191.0, 1601.0, 1787.0), 100.0)
    elsewhere = (2100.0, 1200.0, 2400.0, 1800.0)
    assert not g.is_repeat("motorcycle", elsewhere, 100.5)
    print("OK a_different_vehicle_elsewhere_in_the_zone_is_still_captured")


def test_the_ledger_forgets():
    g = gate(window=3.0)
    box = (1294.0, 1191.0, 1601.0, 1787.0)
    g.record("motorcycle", box, 100.0)
    assert g.is_repeat("motorcycle", box, 102.9)
    assert not g.is_repeat("motorcycle", box, 103.1)
    print("OK the_ledger_forgets")


def test_a_rider_is_not_swallowed_by_the_motorcycle_under_them():
    """Person and motorcycle are captured for different reasons, so separately.

    The rider's box is nested inside the machine's every bit as much as a
    duplicate detection would be. Keeping the ledger per class is what stops a
    face sighting disappearing into the bike it was sitting on.
    """
    g = gate()
    bike = (1294.0, 1191.0, 1601.0, 1787.0)
    rider = (1350.0, 1200.0, 1520.0, 1500.0)
    assert containment(rider, bike) == 1.0  # nested by geometry alone
    g.record("motorcycle", bike, 100.0)
    assert not g.is_repeat("person", rider, 100.0)
    print("OK a_rider_is_not_swallowed_by_the_motorcycle_under_them")


def test_a_capture_that_was_dropped_is_not_remembered():
    """record() is separate from is_repeat() so a refused job can be retried."""
    g = gate()
    box = (1294.0, 1191.0, 1601.0, 1787.0)
    assert not g.is_repeat("motorcycle", box, 100.0)   # asked, and allowed
    # ... the queue refused the job, so nothing is recorded ...
    assert not g.is_repeat("motorcycle", box, 100.1)   # and the retry still is
    print("OK a_capture_that_was_dropped_is_not_remembered")


def test_a_zero_window_turns_the_ledger_off():
    g = gate(window=0.0)
    box = (1294.0, 1191.0, 1601.0, 1787.0)
    g.record("motorcycle", box, 100.0)
    assert not g.is_repeat("motorcycle", box, 100.0)
    print("OK a_zero_window_turns_the_ledger_off")


def test_the_ledger_does_not_grow_without_bound():
    """A busy hour must not leave every capture of it in memory."""
    g = gate(window=3.0)
    for i in range(500):
        box = (1000.0 + i, 1300.0, 1100.0 + i, 1400.0)
        g.is_repeat("motorcycle", box, 100.0 + i)
        g.record("motorcycle", box, 100.0 + i)
    assert len(g._recent["motorcycle"]) <= 4, len(g._recent["motorcycle"])
    print("OK the_ledger_does_not_grow_without_bound")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall capture gate tests passed")
