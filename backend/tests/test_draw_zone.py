"""Tests for holding the live overlay to the ANPR zone.

On a camera pointed down a street most of what YOLO finds is traffic the source
is not being watched for, and its boxes are most of what ends up on screen.
DRAW_ONLY_IN_ZONE holds the overlay to the zone instead.

The properties worth pinning down are the ones that would be easy to get wrong:

* a vehicle *arriving* at the zone is still drawn -- that is the frame an
  operator wants -- so the test is "touches", not "is inside";
* nothing else changes. This hides boxes; it does not stop detecting, counting
  or capturing, and with no zone drawn it does nothing at all.

Run from the backend/ directory:  python -m tests.test_draw_zone
"""
import numpy as np

import app.detection.worker as worker
from app.detection.detector import Detection
from app.detection.worker import _touches_zone


FRAME_H, FRAME_W = 720, 1280
# The zone from the screenshot that prompted this: a band across the lower
# middle of the alley, with the car parked well above it.
ALLEY = {"a": [0.09, 0.48], "b": [0.73, 0.98]}


def frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def zone():
    return worker._zone_px(ALLEY, FRAME_W, FRAME_H)


def car_up_the_street():
    """The car in the screenshot: boxed, and nowhere near the zone."""
    return Detection(476, "car", 0.9, 540, 90, 640, 170)


def bike_in_the_zone():
    return Detection(12, "motorcycle", 0.9, 500, 400, 580, 500)


def bike_arriving():
    """Straddling the zone's upper edge -- half in, half out."""
    top = int(0.48 * FRAME_H)
    return Detection(13, "motorcycle", 0.9, 500, top - 40, 580, top + 40)


def drawn(detections, faces=(), zone_cfg=ALLEY, only_in_zone=True):
    """Draw for real, and report which boxes actually reached the canvas.

    _draw has no return channel for "what did you draw", so the rectangle calls
    are intercepted instead. Re-deriving the answer from the same filter the
    code uses would only prove the filter agrees with itself; this proves _draw
    calls it at all.
    """
    painted = []
    real_rect = worker.cv2.rectangle

    def spy(img, p1, p2, color, thickness, *a, **kw):
        painted.append((p1, p2))
        return real_rect(img, p1, p2, color, thickness, *a, **kw)

    original = worker.settings.draw_only_in_zone
    worker.settings.draw_only_in_zone = only_in_zone
    worker.cv2.rectangle = spy
    try:
        worker._draw(
            frame(), list(detections), None, {}, {"alpr_zone": zone_cfg}, {}, list(faces)
        )
    finally:
        worker.cv2.rectangle = real_rect
        worker.settings.draw_only_in_zone = original

    ids = [
        d.track_id for d in detections
        if ((int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))) in painted
    ]
    names = [
        f[4] for f in faces
        if ((f[0], f[1]), (f[0] + f[2], f[1] + f[3])) in painted
    ]
    return ids, names


def test_a_box_inside_the_zone_touches_it():
    d = bike_in_the_zone()
    assert _touches_zone((d.x1, d.y1, d.x2, d.y2), zone())
    print("OK a_box_inside_the_zone_touches_it")


def test_the_car_up_the_street_does_not():
    """The box the screenshot was complaining about."""
    d = car_up_the_street()
    assert not _touches_zone((d.x1, d.y1, d.x2, d.y2), zone())
    print("OK the_car_up_the_street_does_not")


def test_a_vehicle_arriving_at_the_edge_is_still_drawn():
    """Half in, half out: the frame before it is inside is the one to see."""
    d = bike_arriving()
    assert _touches_zone((d.x1, d.y1, d.x2, d.y2), zone())
    print("OK a_vehicle_arriving_at_the_edge_is_still_drawn")


def test_a_box_merely_alongside_the_zone_is_not_drawn():
    beside = Detection(14, "car", 0.9, 5, 400, 90, 500)  # left of the zone
    assert not _touches_zone((beside.x1, beside.y1, beside.x2, beside.y2), zone())
    print("OK a_box_merely_alongside_the_zone_is_not_drawn")


def test_the_overlay_keeps_only_what_meets_the_zone():
    ids, _ = drawn([car_up_the_street(), bike_in_the_zone(), bike_arriving()])
    assert ids == [12, 13], ids
    print("OK the_overlay_keeps_only_what_meets_the_zone: dropped the car")


def test_faces_follow_the_same_rule():
    in_zone = (500, 400, 40, 40, "BUDI", 0.9)
    up_the_street = (560, 100, 40, 40, "SITI", 0.9)
    _, names = drawn([], [in_zone, up_the_street])
    assert names == ["BUDI"], names
    print("OK faces_follow_the_same_rule")


def test_no_zone_drawn_changes_nothing():
    ids, _ = drawn([car_up_the_street(), bike_in_the_zone()], zone_cfg=None)
    assert ids == [476, 12], ids
    print("OK no_zone_drawn_changes_nothing")


def test_the_setting_turns_it_off():
    ids, _ = drawn([car_up_the_street(), bike_in_the_zone()], only_in_zone=False)
    assert ids == [476, 12], ids
    print("OK the_setting_turns_it_off")


def test_drawing_does_not_mutate_the_caller_list():
    """The filter rebinds a local; the detection loop's list must survive it."""
    dets = [car_up_the_street(), bike_in_the_zone()]
    original = worker.settings.draw_only_in_zone
    worker.settings.draw_only_in_zone = True
    try:
        worker._draw(frame(), dets, None, {}, {"alpr_zone": ALLEY}, {}, [])
    finally:
        worker.settings.draw_only_in_zone = original
    assert len(dets) == 2, dets
    print("OK drawing_does_not_mutate_the_caller_list")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall draw-zone tests passed")
