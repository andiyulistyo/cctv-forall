"""Tests for the plate-reading logic in app.detection.alpr.

Everything here runs without EasyOCR and without a GPU: the pure functions are
called directly, and the pipeline tests drive ALPR through a stub reader that
replays canned OCR output. What is under test is the reasoning applied to that
output -- how boxes become lines, how a misread character is repaired, and
which readings are trusted enough to store.

The canned output is not invented. It is what EasyOCR actually returned for the
four sample motorcycle captures this work was measured on, so a regression here
is a regression on real frames.

Run from the backend/ directory:  python -m tests.test_plate_ocr
"""
import numpy as np

from app.detection.alpr import (
    ALPR, _text_lines, looks_like_plate, normalize_plate, repair_plate,
)


def box(x1, y1, x2, y2):
    """An EasyOCR bounding box: four corners, clockwise from top left."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# --- normalize / looks_like_plate -------------------------------------------

def test_normalize_strips_everything_that_is_not_a_plate_character():
    assert normalize_plate("b 6084 txb") == "B6084TXB"
    assert normalize_plate("B-6084.TXB") == "B6084TXB"
    print("OK normalize_strips_everything_that_is_not_a_plate_character")


def test_looks_like_plate_accepts_the_indonesian_shapes():
    assert looks_like_plate("B6084TXB")   # Jakarta, 4 digits, 3 suffix
    assert looks_like_plate("AD12AB")     # two-letter area, 2 digits
    assert not looks_like_plate("0829")   # the expiry stamp, not a plate
    assert not looks_like_plate("B6084TXB0829")  # plate with expiry glued on
    print("OK looks_like_plate_accepts_the_indonesian_shapes")


# --- repair_plate -----------------------------------------------------------

def test_a_clean_plate_is_returned_unchanged_and_free():
    assert repair_plate("B6084TXB") == ("B6084TXB", 0)
    print("OK a_clean_plate_is_returned_unchanged_and_free")


def test_letters_in_the_digit_block_are_repaired_to_digits():
    """"B6O84TXB" is a plate whose 0 came back as an O."""
    assert repair_plate("B6O84TXB") == ("B6084TXB", 1)
    print("OK letters_in_the_digit_block_are_repaired_to_digits")


def test_digits_in_the_suffix_block_are_repaired_to_letters():
    """A "6" cannot be a suffix character, so it is the G it looks like."""
    assert repair_plate("B6084TX6") == ("B6084TXG", 1)
    print("OK digits_in_the_suffix_block_are_repaired_to_letters")


def test_repair_prefers_the_split_needing_fewest_substitutions():
    """"B6084TXB" parses as 1+4+3 free, so no costlier split may win."""
    plate, cost = repair_plate("B6084TXB")
    assert (plate, cost) == ("B6084TXB", 0), (plate, cost)
    print("OK repair_prefers_the_split_needing_fewest_substitutions")


def test_a_string_with_no_plausible_split_is_refused():
    assert repair_plate("XX") is None            # too short
    assert repair_plate("ABCDEFGHIJ") is None    # too long
    assert repair_plate("ABCDEFGH") is None      # no digits anywhere
    print("OK a_string_with_no_plausible_split_is_refused")


def test_l_is_never_guessed_at():
    """L is left alone on purpose.

    On sample 3 the character EasyOCR reported as "L" was a 4, and on other
    plates the same shape is a 1. A map that had to choose would be wrong about
    half the time, so the digit block simply refuses an L and the reading is
    thrown out rather than invented.
    """
    assert repair_plate("BL893UBY") == ("BL893UBY", 0)  # valid as-is: BL area
    assert repair_plate("BLLLLUBY") is None             # not rescued into digits
    print("OK l_is_never_guessed_at")


# --- _text_lines ------------------------------------------------------------

def test_a_plate_split_across_boxes_is_joined_left_to_right():
    """The motorcycle failure: wide gaps make EasyOCR report three boxes.

    None of them is a plate alone, which is exactly why reading boxes
    individually threw the plate away.
    """
    results = [
        (box(60, 10, 80, 40), "B", 0.9),
        (box(100, 10, 180, 40), "6084", 0.8),
        (box(200, 10, 280, 40), "TXB", 0.7),
    ]
    lines = _text_lines(results)
    assert len(lines) == 1, lines
    text, conf, _bbox = lines[0]
    assert text == "B6084TXB", text
    # Confidence is weighted by how many characters each box contributed, so
    # the four-digit box counts for more than the single letter.
    assert 0.7 < conf < 0.85, conf
    print("OK a_plate_split_across_boxes_is_joined_left_to_right")


def test_the_expiry_stamp_stays_on_its_own_line():
    """Joining everything would produce "B6084TXB0829" and fail the length
    check -- which is how the number was lost on the tight sample crops."""
    results = [
        (box(60, 10, 280, 40), "B 6084 TXB", 0.9),
        (box(180, 60, 260, 85), "08 29", 0.95),
    ]
    lines = _text_lines(results)
    assert [t for t, _c, _b in lines] == ["B6084TXB", "0829"], lines
    print("OK the_expiry_stamp_stays_on_its_own_line")


def test_boxes_are_ordered_by_position_not_by_detection_order():
    results = [
        (box(200, 10, 280, 40), "TXB", 0.7),
        (box(60, 10, 80, 40), "B", 0.9),
        (box(100, 10, 180, 40), "6084", 0.8),
    ]
    assert _text_lines(results)[0][0] == "B6084TXB"
    print("OK boxes_are_ordered_by_position_not_by_detection_order")


# --- the acceptance rules ---------------------------------------------------

class FakeReader:
    """Replays one canned OCR result per readtext call.

    Once the canned passes run out it returns nothing, so a test controls
    exactly how many variants saw a given plate -- which is the input the
    agreement rule is judged on.
    """

    def __init__(self, *passes):
        self._passes = list(passes)
        self.calls = 0

    def readtext(self, image, **kw):
        self.calls += 1
        if self.calls <= len(self._passes):
            return self._passes[self.calls - 1]
        return []


def alpr_with(*passes, **kw):
    """An ALPR wired to a FakeReader, with EasyOCR never imported."""
    a = ALPR.__new__(ALPR)
    a._reader = FakeReader(*passes)
    a._plate_detector = None
    a._available = True
    a.device = "cpu"
    a.plate_imgsz = 320
    a.plate_conf = 0.25
    a.min_confidence = kw.get("min_confidence", 0.20)
    a.ocr_min_height = 64
    a.ocr_min_width = 240
    a.good_enough = kw.get("good_enough", 0.75)
    a.max_passes = kw.get("max_passes", 8)
    a.half = False
    return a


# A crop big enough that _plate_rois offers the lower band as well.
CROP = np.full((200, 400, 3), 128, dtype=np.uint8)

CONFIDENT = [(box(60, 10, 280, 40), "B 6084 TXB", 0.85)]


def test_a_confident_plate_is_read_and_stops_the_ladder():
    a = alpr_with(CONFIDENT)
    result = a.read_plate(CROP)
    assert result is not None
    text, conf, _img = result
    assert text == "B6084TXB", text
    # One pass: a plate this confident has nothing to gain from more variants.
    assert a._reader.calls == 1, a._reader.calls
    print("OK a_confident_plate_is_read_and_stops_the_ladder")


def test_a_split_plate_is_read_where_per_box_scoring_could_not():
    """The regression this whole change exists for."""
    a = alpr_with([
        (box(60, 10, 80, 40), "B", 0.88),
        (box(100, 10, 180, 40), "6084", 0.86),
        (box(200, 10, 280, 40), "TXB", 0.84),
    ])
    result = a.read_plate(CROP)
    assert result is not None and result[0] == "B6084TXB", result
    print("OK a_split_plate_is_read_where_per_box_scoring_could_not")


def test_the_expiry_stamp_is_never_returned_as_the_plate():
    """What the old lower-band ROI actually produced on the tight crops."""
    a = alpr_with([(box(180, 60, 260, 85), "08 29", 0.95)])
    assert a.read_plate(CROP) is None
    print("OK the_expiry_stamp_is_never_returned_as_the_plate")


def test_one_variant_inventing_a_plate_is_not_trusted():
    """Sample 1: an illegible plate whose every variant produced a *different*
    well-formed string at 0.21-0.26. Under a flat floor the top one would have
    been stored as fact."""
    a = alpr_with(
        [(box(0, 0, 40, 20), "S474N", 0.23)],
        [(box(0, 0, 40, 20), "S478D", 0.21)],
        [(box(0, 0, 40, 20), "S47B", 0.26)],
    )
    assert a.read_plate(CROP) is None
    print("OK one_variant_inventing_a_plate_is_not_trusted")


def test_agreement_lowers_the_bar_it_does_not_remove_it():
    """Sample 4: two variants agreeing on a wrong plate at 0.23.

    Agreement is evidence, not proof. Two variants take something off the bar
    an uncorroborated read has to clear, but not enough that 0.23 gets stored;
    only once the reading survives most of the ladder does it fall back to the
    configured floor, which is the point at which a weak score is worth having.
    """
    weak = [(box(0, 0, 40, 20), "B 7330 TRU", 0.23)]
    assert alpr_with(weak).read_plate(CROP) is None           # 1 variant
    assert alpr_with(weak, weak).read_plate(CROP) is None     # 2 variants
    many = alpr_with(weak, weak, weak, weak).read_plate(CROP)  # 4 variants
    assert many is not None and many[0] == "B7330TRU", many
    print("OK agreement_lowers_the_bar_it_does_not_remove_it")


def test_a_corroborated_middling_read_is_accepted():
    """A real plate several distortions of the crop agree on is worth having
    even when no single pass was certain of it."""
    mid = [(box(0, 0, 200, 40), "B 6084 TXB", 0.50)]
    result = alpr_with(mid, mid).read_plate(CROP)
    assert result is not None and result[0] == "B6084TXB", result
    print("OK a_corroborated_middling_read_is_accepted")


def test_a_one_digit_plate_is_refused_however_well_formed():
    """"G6JJO" satisfies the plate pattern exactly and is a tail light."""
    a = alpr_with([(box(0, 0, 40, 20), "G6JJO", 0.9)])
    assert a.read_plate(CROP) is None
    print("OK a_one_digit_plate_is_refused_however_well_formed")


def test_the_pass_budget_is_never_exceeded():
    """An unreadable crop must not grind through every stage: the plate queue
    is short, and time spent here is other vehicles' turn."""
    a = alpr_with([(box(0, 0, 40, 20), "??", 0.05)], max_passes=3)
    assert a.read_plate(CROP) is None
    assert a._reader.calls <= 3, a._reader.calls
    print("OK the_pass_budget_is_never_exceeded")


def test_an_empty_or_missing_crop_is_handled():
    a = alpr_with(CONFIDENT)
    assert a.read_plate(None) is None
    assert a.read_plate(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    print("OK an_empty_or_missing_crop_is_handled")


def test_a_reader_that_raises_does_not_escape():
    """OCR failing is a missed plate, not a dead worker thread."""
    class Boom:
        def readtext(self, image, **kw):
            raise RuntimeError("cuda oom")

    a = alpr_with(CONFIDENT)
    a._reader = Boom()
    assert a.read_plate(CROP) is None
    print("OK a_reader_that_raises_does_not_escape")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} plate-OCR tests passed.")
