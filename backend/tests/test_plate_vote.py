"""Tests for the cross-frame plate vote in app.detection.plate_vote.

What this pins down is the change of mind behind the module: a plate is decided
by every frame that read it, not by the single frame that scored highest. The
cases below are the ones that separate the two -- a confident minority, a
truncated reading that must not drag the columns out of line, and a plate no
single frame ever read correctly but the frames together did.

Stdlib only, like the module under test.

Run from the backend/ directory:  python -m tests.test_plate_vote
"""
from app.detection.plate_vote import PlateVoter, plate_shape


def voter(*readings) -> PlateVoter:
    v = PlateVoter()
    for text, conf in readings:
        v.add(text, conf)
    return v


# --- shape ------------------------------------------------------------------

def test_shape_splits_the_three_blocks():
    assert plate_shape("B6084TXB") == (1, 4, 3)
    assert plate_shape("AD12AB") == (2, 2, 2)
    print("OK shape_splits_the_three_blocks")


def test_unparseable_strings_group_by_length_and_never_collide_with_a_real_shape():
    assert plate_shape("????") == (-1, 4, -1)
    assert plate_shape("????") != plate_shape("?????")
    # (-1, ...) cannot be produced by a real plate: the blocks are all >= 1.
    assert plate_shape("????")[0] == -1
    print("OK unparseable_strings_group_by_length")


# --- the case argmax got wrong ----------------------------------------------

def test_agreement_beats_a_single_more_confident_read():
    """Eight frames at 0.70 against one at 0.72. The eight are right."""
    v = voter(*[("B1234XYZ", 0.70)] * 8, ("B1234XY2", 0.72))
    assert v.text == "B1234XYZ", v.summary()
    assert v.votes == 8
    print(f"OK agreement_beats_a_single_more_confident_read: {v.summary()}")


def test_one_read_is_simply_that_read():
    v = voter(("B6084TXB", 0.61))
    assert (v.text, v.confidence) == ("B6084TXB", 0.61)
    print("OK one_read_is_simply_that_read")


def test_a_lone_confident_read_still_wins_against_nothing_much():
    """Agreement is not a majority rule: two faint misreads that do not even
    agree with *each other* must not outvote one clean read."""
    v = voter(("B6084TXB", 0.88), ("B6084TX8", 0.10), ("B6O84TXB", 0.09))
    assert v.text == "B6084TXB", v.summary()
    print(f"OK a_lone_confident_read_still_wins: {v.summary()}")


# --- alignment --------------------------------------------------------------

def test_a_truncated_reading_cannot_shift_the_columns():
    """A frame that missed the area letter reads "1234XYZ". Voting that
    against "B1234XYZ" column by column would put "1" under "B" and score
    every position wrong; the shapes are kept apart instead."""
    v = voter(("B1234XYZ", 0.60), ("B1234XYZ", 0.55), ("1234XYZ", 0.95))
    assert v.text == "B1234XYZ", v.summary()
    print(f"OK a_truncated_reading_cannot_shift_the_columns: {v.summary()}")


def test_the_shape_with_the_most_confidence_behind_it_wins():
    """...and that is total confidence, not head count: one clear full read
    outweighs three faint truncated ones."""
    v = voter(("AD12AB", 0.90), ("D12AB", 0.20), ("D12AB", 0.20), ("D12AB", 0.20))
    assert v.text == "AD12AB", v.summary()
    print(f"OK the_shape_with_the_most_confidence_behind_it_wins: {v.summary()}")


# --- a plate nobody read whole ----------------------------------------------

# Five readings of one motorcycle. Two frames fumble the leading letter, three
# fumble the trailing one, and they fumble it differently each time -- so every
# frame is wrong, no two frames are wrong the same way, and every *column* still
# has a clear majority. This is the shape of the whole argument for voting: the
# plate is right there in the readings and no single reading contains it.
_NOBODY_GOT_IT_WHOLE = (
    ("86084TXB", 0.62),
    ("36084TXB", 0.58),
    ("B6084TX8", 0.66),
    ("B6084TX9", 0.55),
    ("B6084TX0", 0.60),
)


def test_a_consensus_can_be_assembled_from_columns_no_frame_got_right_together():
    v = voter(*_NOBODY_GOT_IT_WHOLE)
    assert v.text == "B6084TXB", v.summary()
    assert v.text not in {t for t, _ in _NOBODY_GOT_IT_WHOLE}
    print(f"OK a_consensus_can_be_assembled_from_columns: {v.summary()}")


def test_an_assembled_consensus_never_claims_more_than_its_best_frame():
    v = voter(*_NOBODY_GOT_IT_WHOLE)
    assert v.text not in {r.text for r in v.readings}, "this case is about a new string"
    assert v.confidence <= max(r.conf for r in v.readings), v.summary()
    print(f"OK an_assembled_consensus_never_claims_more_than_its_best_frame: {v.confidence:.2f}")


def test_confidence_stays_on_the_scale_a_single_read_reports():
    """ALPR_MIN_CONFIDENCE, the listing filter and every row written before the
    vote existed all read this number. It has to keep meaning the same thing."""
    v = voter(("B1234XYZ", 0.42), ("B1234XYZ", 0.81), ("B1234XYZ", 0.55))
    assert v.confidence == 0.81, v.summary()
    print("OK confidence_stays_on_the_scale_a_single_read_reports")


# --- bookkeeping ------------------------------------------------------------

def test_add_reports_whether_the_answer_moved():
    v = PlateVoter()
    assert v.add("B1234XYZ", 0.60) is True      # nothing -> something
    assert v.add("B1234XYZ", 0.50) is False     # same answer, weaker read
    assert v.add("B1234XYZ", 0.90) is True      # same answer, better number
    assert v.add("", 0.99) is False             # an empty read is not a read
    print("OK add_reports_whether_the_answer_moved")


def test_the_answer_does_not_depend_on_the_order_the_frames_arrived_in():
    reads = [("B6084TXB", 0.61), ("B6O84TXB", 0.61), ("B6084TXB", 0.44), ("BG084TXB", 0.61)]
    first = voter(*reads)
    second = voter(*reversed(reads))
    assert (first.text, first.confidence) == (second.text, second.confidence), (
        f"{first.summary()} != {second.summary()}"
    )
    print(f"OK the_answer_does_not_depend_on_order: {first.summary()}")


def test_readings_are_capped_so_a_vehicle_that_never_leaves_cannot_grow_forever():
    v = PlateVoter()
    for _ in range(500):
        v.add("B1234XYZ", 0.6)
    assert len(v.readings) <= 64, len(v.readings)
    assert v.text == "B1234XYZ"
    print(f"OK readings_are_capped: {len(v.readings)} kept of 500")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} plate-vote tests passed.")
