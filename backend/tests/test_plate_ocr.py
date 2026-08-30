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
import tempfile
from pathlib import Path

import numpy as np

from app.detection.alpr import (
    ALPR, _text_lines, _upscale, looks_like_plate, normalize_plate,
    recogniser_kwargs, repair_plate,
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


# --- enlarging the crop before OCR ------------------------------------------
#
# The sizes here are the ones measured off the reviewed reads, not invented.
# What this section guards is the failure that hid in plain sight for a while:
# the 1.5x floor, meant to skip a pointless marginal resize, was instead
# swallowing the median plate whole.

def test_the_median_real_crop_is_actually_enlarged():
    """172x74 is the median plate crop off these cameras.

    Against the old 64/240 targets it asked for 240/172 = 1.40x, fell under
    the 1.5x floor and went to OCR untouched -- as did 57% of the reviewed
    crops. Half of all reads were coming back missing characters.
    """
    crop = np.zeros((74, 172), dtype=np.uint8)
    out = _upscale(crop, 160, 600)
    assert out.shape[0] > 74 and out.shape[1] > 172, out.shape
    # Old targets, same crop: the regression this pins.
    assert _upscale(crop, 64, 240).shape == crop.shape


def test_the_width_term_is_what_binds_on_a_plate():
    """A plate is wide and short, so 600/w outruns 160/h and sets the scale.

    Which is why the targets survive a change of camera distance: whether the
    crop arrives 172 px or 226 px wide, it leaves about 600 px across.
    """
    for w, h in ((172, 74), (226, 92), (120, 60)):
        out = _upscale(np.zeros((h, w), dtype=np.uint8), 160, 600)
        assert abs(out.shape[1] - 600) <= 6, (w, h, out.shape)


def test_a_crop_that_is_already_big_enough_is_left_alone():
    """The floor still has a job -- it is just no longer doing it to plates."""
    big = np.zeros((300, 900), dtype=np.uint8)
    assert _upscale(big, 160, 600).shape == big.shape


def test_enlargement_is_still_capped():
    """Past the cap this is inventing detail, not revealing it."""
    tiny = np.zeros((20, 40), dtype=np.uint8)
    out = _upscale(tiny, 160, 600, cap=6.0)
    assert out.shape[1] <= 40 * 6, out.shape


# --- loading a custom recogniser --------------------------------------------
#
# This is the slot a recogniser fine-tuned on the exported dataset plugs into.
# Nothing here loads a model: what is worth pinning is the decision made before
# one is loaded, because getting it wrong takes ANPR down rather than merely
# leaving it less accurate.

def test_no_recogniser_configured_asks_easyocr_for_nothing():
    """The default has to stay the stock reader, with no directories invented."""
    kwargs, missing = recogniser_kwargs("", "/models", "/nets")
    assert kwargs == {} and missing == []
    # "standard" is spelled out in .env files as much as left empty.
    assert recogniser_kwargs("standard", "/models", "/nets") == ({}, [])
    print("OK no_recogniser_configured_asks_for_nothing")


def test_easyocrs_own_networks_are_named_but_never_looked_for_on_disk():
    """EasyOCR downloads these itself; checking a local path would reject them."""
    kwargs, missing = recogniser_kwargs("english_g2", "/nonexistent", "/nonexistent")
    assert kwargs == {"recog_network": "english_g2"}
    assert missing == []
    print("OK stock_networks_are_not_looked_for_on_disk")


def test_a_custom_recogniser_names_all_three_files_it_needs():
    """A fine-tune is three files, and it is usually the .py that got left out.

    EasyOCR stops at the first one it cannot open, so reporting them one at a
    time would cost a restart per file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "plate_id_v1.pth").write_bytes(b"")
        kwargs, missing = recogniser_kwargs("plate_id_v1", tmp, tmp)
        assert kwargs["recog_network"] == "plate_id_v1"
        assert kwargs["model_storage_directory"] == tmp
        assert kwargs["user_network_directory"] == tmp
        assert len(missing) == 2, missing
        assert any(m.endswith("plate_id_v1.yaml") for m in missing)
        assert any(m.endswith("plate_id_v1.py") for m in missing)
    print("OK a_custom_recogniser_names_all_three: both reported at once")


def test_a_complete_recogniser_reports_nothing_missing():
    with tempfile.TemporaryDirectory() as tmp:
        for ext in ("pth", "yaml", "py"):
            (Path(tmp) / f"plate_id_v1.{ext}").write_bytes(b"")
        kwargs, missing = recogniser_kwargs("plate_id_v1", tmp, tmp)
        assert missing == [], missing
        assert kwargs["recog_network"] == "plate_id_v1"
    print("OK a_complete_recogniser_reports_nothing_missing")


def test_the_architecture_can_live_apart_from_the_weights():
    """EasyOCR keeps the two directories separate; so does the config."""
    with tempfile.TemporaryDirectory() as tmp:
        models, nets = Path(tmp) / "weights", Path(tmp) / "user_network"
        models.mkdir()
        nets.mkdir()
        (models / "plate_id_v1.pth").write_bytes(b"")
        (nets / "plate_id_v1.yaml").write_bytes(b"")
        (nets / "plate_id_v1.py").write_bytes(b"")
        kwargs, missing = recogniser_kwargs("plate_id_v1", str(models), str(nets))
        assert missing == [], missing
        assert kwargs["model_storage_directory"] == str(models)
        assert kwargs["user_network_directory"] == str(nets)
    print("OK the_architecture_can_live_apart_from_the_weights")


def test_an_empty_network_dir_falls_back_to_the_weights_dir():
    """One folder holding all three files is how a fine-tune usually arrives."""
    with tempfile.TemporaryDirectory() as tmp:
        for ext in ("pth", "yaml", "py"):
            (Path(tmp) / f"plate_id_v1.{ext}").write_bytes(b"")
        kwargs, missing = recogniser_kwargs("plate_id_v1", tmp, "")
        assert missing == [], missing
        assert kwargs["user_network_directory"] == tmp
    print("OK an_empty_network_dir_falls_back_to_the_weights_dir")


def test_a_missing_recogniser_is_reported_not_raised():
    """The whole reason this check exists rather than letting EasyOCR raise.

    A mistyped OCR_RECOG_NETWORK should cost accuracy for a day, not plate
    reading: the caller drops the kwargs and runs stock. Anything that threw
    here would instead be caught by ALPR.__init__ as "could not init EasyOCR",
    leaving the worker with alpr = None and no plate reading at all.
    """
    with tempfile.TemporaryDirectory() as tmp:
        kwargs, missing = recogniser_kwargs("typo_v1", tmp, tmp)
        assert len(missing) == 3, missing
        # Returned anyway: it is the caller that decides to drop them.
        assert kwargs["recog_network"] == "typo_v1"
    print("OK a_missing_recogniser_is_reported_not_raised")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} plate-OCR tests passed.")
