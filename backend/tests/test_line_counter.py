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


# --- Regressions for the overcounting seen in the field ---------------------
#
# The sample junction feed logged 8.5% more car crossings than there were car
# tracks and 18.5% more for trucks -- ordered by vehicle size -- in a
# characteristic "out,in,out" pattern. These reproduce the two mechanisms
# behind that: box jitter driving the crossing test, and the centroid moving
# for reasons that have nothing to do with the vehicle moving.


# The two candidate reference points, kept in stdlib terms so this file stays
# runnable without numpy/torch. test_detection_agrees_on_reference_points below
# checks they still match Detection's properties when the deps are available.
def _centroid(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _ground(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, y2)


def test_detection_agrees_on_reference_points():
    try:
        from app.detection.detector import Detection
    except ImportError as exc:  # numpy/torch absent: the counter tests still run
        print(f"SKIP detection_agrees_on_reference_points ({exc})")
        return
    d = Detection(1, "truck", 0.9, 600.0, 150.0, 700.0, 420.0)
    assert d.centroid == _centroid(600.0, 150.0, 700.0, 420.0), d.centroid
    assert d.ground_point == _ground(600.0, 150.0, 700.0, 420.0), d.ground_point
    print("OK detection_agrees_on_reference_points:", d.ground_point)


def _counter_720p(hysteresis=0.02):
    """A horizontal line across the middle of a 720p frame."""
    c = LineCounter(line_norm=((0.0, 0.5), (1.0, 0.5)), hysteresis=hysteresis)
    c.set_frame_size(1280, 720)  # line at y = 360
    return c


def test_jitter_near_line_counts_once():
    """The observed out,in,out: a vehicle wobbling as it crosses."""
    path = [(640, y) for y in (330, 348, 366, 355, 370, 359, 372, 390, 420, 470)]

    without = _counter_720p(hysteresis=0.0)
    feed(without, 1, "car", path)
    raw = sum(without.total().get("car", {}).values())

    with_band = _counter_720p()
    feed(with_band, 1, "car", path)
    counted = with_band.total()["car"]

    assert raw > 1, f"expected the jitter to over-count without a band, got {raw}"
    assert sum(counted.values()) == 1, counted
    print(f"OK jitter_near_line_counts_once: tanpa band={raw} -> dengan band=1")


def test_genuine_return_still_counts_both_ways():
    """Hysteresis must not silence a vehicle that really does turn back."""
    c = _counter_720p()
    feed(c, 1, "car", [(640, y) for y in (280, 320, 400, 440, 400, 320, 280)])
    t = c.total()["car"]
    assert t["in"] == 1 and t["out"] == 1, t
    print("OK genuine_return_still_counts_both_ways:", t)


def test_ground_point_survives_box_jitter():
    """A truck whose top edge flickers must not move its reference point.

    The bottom edge advances steadily; the box height alternates as the model
    takes in or leaves out the container. The centroid inherits half of that
    flicker and walks backwards over the line; the ground point does not.
    """
    boxes = []
    for i in range(25):
        y2 = 300.0 + i * 9  # wheels advance steadily
        height = 150.0 if i % 2 == 0 else 210.0  # container in/out
        boxes.append((600.0, y2 - height, 700.0, y2))

    by_centroid = _counter_720p(hysteresis=0.0)
    for box in boxes:
        by_centroid.update([(1, "truck", *_centroid(*box))])

    by_ground = _counter_720p(hysteresis=0.0)
    for box in boxes:
        by_ground.update([(1, "truck", *_ground(*box))])

    n_centroid = sum(by_centroid.total().get("truck", {}).values())
    n_ground = sum(by_ground.total().get("truck", {}).values())
    assert n_centroid > 1, f"expected the centroid to over-count, got {n_centroid}"
    assert n_ground == 1, by_ground.total()
    print(f"OK ground_point_survives_box_jitter: centroid={n_centroid} -> ground={n_ground}")


def test_tall_and_short_vehicles_cross_together():
    """A bus and a motorcycle side by side must cross at the same moment.

    Same position on the road, same speed, different heights. With the centroid
    the bus counts frames earlier, which is a systematic bias between classes,
    not noise.
    """
    def crossing_frame(height, use_ground):
        c = _counter_720p(hysteresis=0.0)
        for i in range(40):
            # +1 so neither reference point ever lands exactly on the line,
            # where "which side" has no answer.
            y2 = 301.0 + i * 6
            box = (600.0, y2 - height, 700.0, y2)
            point = _ground(*box) if use_ground else _centroid(*box)
            if c.update([(1, "bus", *point)]):
                return i
        return None

    bus_c, moto_c = crossing_frame(220, False), crossing_frame(60, False)
    bus_g, moto_g = crossing_frame(220, True), crossing_frame(60, True)
    assert bus_g == moto_g, (bus_g, moto_g)
    assert bus_c != moto_c, (bus_c, moto_c)
    print(
        f"OK tall_and_short_vehicles_cross_together: centroid bus={bus_c} motor={moto_c} "
        f"(beda {abs(bus_c - moto_c)} frame) -> ground bus={bus_g} motor={moto_g}"
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} line-counter tests passed.")
