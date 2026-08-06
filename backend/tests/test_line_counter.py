"""Standalone correctness tests for the line-crossing counter.

Run from the backend/ directory:  python -m tests.test_line_counter
(No heavy deps required — line_counter only uses the stdlib.)
"""
from app.detection.line_counter import LineCounter


def make_counter(a, b, w=100, h=100):
    c = LineCounter(line_norm=(a, b))
    c.set_frame_size(w, h)
    return c


def feed(counter, track_id, cls, path):
    """Feed a sequence of centroids for one track, one per frame."""
    for (x, y) in path:
        counter.update([(track_id, cls, x, y)])


def test_left_to_right_counts_once():
    c = make_counter((0.5, 0.0), (0.5, 1.0))
    feed(c, 1, "car", [(40, 50), (60, 50), (70, 50), (80, 50)])
    totals = c.total()
    assert sum(totals["car"].values()) == 1, totals
    print("OK left_to_right_counts_once:", totals)


def test_direction_is_opposite_for_reverse():
    c1 = make_counter((0.5, 0.0), (0.5, 1.0))
    feed(c1, 1, "car", [(40, 50), (60, 50)])  # left -> right
    d1 = "in" if c1.total()["car"]["in"] else "out"

    c2 = make_counter((0.5, 0.0), (0.5, 1.0))
    feed(c2, 2, "car", [(60, 50), (40, 50)])  # right -> left
    d2 = "in" if c2.total()["car"]["in"] else "out"

    assert d1 != d2, (c1.total(), c2.total())
    print("OK direction_is_opposite_for_reverse:", d1, d2)


def test_no_count_when_staying_one_side():
    c = make_counter((0.5, 0.0), (0.5, 1.0))
    feed(c, 1, "car", [(60, 50), (70, 50), (80, 50)])
    assert c.total().get("car", {"in": 0, "out": 0}) == {"in": 0, "out": 0}, c.total()
    print("OK no_count_when_staying_one_side")


def test_no_double_count_hovering():
    # An object that crosses once then wiggles near the line must count once.
    c = make_counter((0.5, 0.0), (0.5, 1.0))
    feed(c, 1, "car", [(40, 50), (60, 50), (58, 50), (62, 50)])
    assert sum(c.total()["car"].values()) == 3 or sum(c.total()["car"].values()) >= 1
    # Precisely: crossings at 40->60 (yes), 60->58 (no, same side), 58->62 (no).
    assert sum(c.total()["car"].values()) == 1, c.total()
    print("OK no_double_count_hovering:", c.total())


def test_outside_segment_not_counted():
    # Line only covers the top half (y in [0,50]); object crosses at y=80.
    c = make_counter((0.5, 0.0), (0.5, 0.5))
    feed(c, 1, "car", [(40, 80), (60, 80)])
    assert c.total().get("car", {"in": 0, "out": 0}) == {"in": 0, "out": 0}, c.total()
    print("OK outside_segment_not_counted")


def test_multi_class_separate_counts():
    c = make_counter((0.5, 0.0), (0.5, 1.0))
    feed(c, 1, "car", [(40, 30), (60, 30)])
    feed(c, 2, "truck", [(40, 70), (60, 70)])
    feed(c, 3, "truck", [(60, 20), (40, 20)])
    t = c.total()
    assert sum(t["car"].values()) == 1, t
    assert sum(t["truck"].values()) == 2, t
    print("OK multi_class_separate_counts:", t)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} line-counter tests passed.")
