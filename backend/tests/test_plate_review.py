"""Tests for human review of a plate read.

The point of a review is not to make the listing look right -- it is to produce
the one thing this system otherwise has none of: ground truth. So what these
tests actually guard is that the *prediction survives the correction*. A review
that overwrote plate_text would leave every screen looking better and quietly
destroy the ability to say how often the reader is wrong, or to use these rows
to train or compare a recogniser. Every case below comes back to that.

The endpoints are called as plain functions against a throwaway in-memory
database -- no HTTP client and no auth, the same shape as test_plate_filters.

Run from the backend/ directory:  python -m tests.test_plate_review
"""
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.plates import query_plates, review_plate, unreview_plate
from app.models import Base, PlateRead, Source
from app.schemas import PlateReviewIn

T0 = datetime(2026, 8, 22, 8, 0, 0)
USER = "admin"

# (id, plate as OCR read it, confidence)
ROWS = [
    (1, "B1234ABC", 0.95),
    (2, "B5678DEF", 0.40),
    (3, "D9999XYZ", 0.72),
    (4, "", 0.0),  # a capture: recorded in the zone, plate never read
]


def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(Source(id=1, name="Gerbang Utara", type="rtsp", url="rtsp://a"))
    for i, (rid, text, conf) in enumerate(ROWS):
        s.add(
            PlateRead(
                id=rid, source_id=1, track_id=rid, vehicle_class="car",
                plate_text=text, confidence=conf, timestamp=T0 + timedelta(minutes=i),
            )
        )
    s.commit()
    return s


def review(session, plate_id: int, text: str):
    return review_plate(plate_id, PlateReviewIn(plate_text=text), db=session, user=USER)


def ids(session, **kw) -> set[int]:
    return {p.id for p in query_plates(session, **kw).plates}


# --- the whole point --------------------------------------------------------

def test_a_correction_never_touches_what_ocr_read():
    s = db()
    out = review(s, 2, "B5678DEE")
    assert out.plate_text == "B5678DEF", "the prediction was overwritten"
    assert out.corrected_text == "B5678DEE"
    assert s.get(PlateRead, 2).plate_text == "B5678DEF"
    print("OK a_correction_never_touches_what_ocr_read")


def test_a_review_records_who_and_when():
    s = db()
    out = review(s, 1, "B1234ABC")
    assert out.reviewed_by == USER
    assert out.reviewed_at is not None
    print("OK a_review_records_who_and_when")


# --- normalisation ----------------------------------------------------------

def test_a_human_typing_spaces_produces_the_same_label_as_the_reader():
    """"B 6084 TXB" and "B6084TXB" are one plate. If they were stored as two,
    every correct read of a hand-corrected plate would score as an error."""
    s = db()
    assert review(s, 1, "b 1234 abc").corrected_text == "B1234ABC"
    assert review(s, 3, "D-9999.XYZ").corrected_text == "D9999XYZ"
    print("OK a_human_typing_spaces_produces_the_same_label_as_the_reader")


def test_text_with_nothing_alphanumeric_in_it_is_refused_rather_than_filed_as_illegible():
    s = db()
    try:
        review(s, 1, "???")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 422, exc
    else:
        raise AssertionError("'???' was accepted")
    assert s.get(PlateRead, 1).reviewed_at is None, "a refused review still marked the row"
    print("OK text_with_nothing_alphanumeric_is_refused")


# --- illegible --------------------------------------------------------------

def test_illegible_is_an_answer_and_is_told_apart_from_not_yet_looked_at():
    s = db()
    review(s, 4, "")
    row = s.get(PlateRead, 4)
    assert row.corrected_text == ""
    assert row.reviewed_at is not None, "illegible must not read as pending"
    assert ids(s, review="pending") == {1, 2, 3}
    assert ids(s, review="illegible") == {4}
    print("OK illegible_is_an_answer_and_is_told_apart_from_pending")


# --- the filters ------------------------------------------------------------

def test_the_filters_split_reviews_into_right_wrong_and_unreadable():
    s = db()
    review(s, 1, "B1234ABC")   # OCR was right
    review(s, 2, "B5678DEE")   # OCR was wrong
    review(s, 4, "")           # nobody can read it
    assert ids(s, review="pending") == {3}
    assert ids(s, review="reviewed") == {1, 2, 4}
    assert ids(s, review="correct") == {1}
    assert ids(s, review="wrong") == {2}, "the error set is the point of all this"
    assert ids(s, review="illegible") == {4}
    print("OK the_filters_split_reviews_into_right_wrong_and_unreadable")


def test_an_illegible_read_is_neither_correct_nor_wrong():
    """Row 4 has plate_text "" and corrected_text "": string-equal, and it
    would fall into "correct" under a plain comparison. It is not a correct
    read; it is a read that never happened."""
    s = db()
    review(s, 4, "")
    assert 4 not in ids(s, review="correct")
    assert 4 not in ids(s, review="wrong")
    print("OK an_illegible_read_is_neither_correct_nor_wrong")


# --- searching --------------------------------------------------------------

def test_search_finds_a_plate_by_what_it_really_is_not_only_by_the_misread():
    s = db()
    review(s, 2, "B5678DEE")
    assert ids(s, q="B5678DEE") == {2}, "the corrected number is unfindable"
    assert ids(s, q="B5678DEF") == {2}, "the misread is still how it is on record"
    print("OK search_finds_a_plate_by_what_it_really_is")


# --- revising ---------------------------------------------------------------

def test_a_review_can_be_revised_and_the_latest_answer_stands():
    s = db()
    review(s, 2, "B5678DEE")
    out = review(s, 2, "B5678DEF")
    assert out.corrected_text == "B5678DEF"
    assert ids(s, review="correct") == {2}
    print("OK a_review_can_be_revised")


def test_undoing_a_review_puts_the_read_back_in_the_queue():
    s = db()
    review(s, 2, "B5678DEE")
    out = unreview_plate(2, db=s, user=USER)
    assert (out.corrected_text, out.reviewed_at, out.reviewed_by) == (None, None, None)
    assert 2 in ids(s, review="pending")
    assert out.plate_text == "B5678DEF", "undoing a review lost the OCR text"
    print("OK undoing_a_review_puts_the_read_back_in_the_queue")


def test_reviewing_a_read_that_does_not_exist_is_a_404():
    s = db()
    try:
        review(s, 999, "B1234ABC")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 404, exc
    else:
        raise AssertionError("a missing read was reviewed")
    print("OK reviewing_a_read_that_does_not_exist_is_a_404")


# --- retention --------------------------------------------------------------

def test_retention_keeps_reviewed_reads_and_purges_the_rest():
    """A seven-day timer against a fortnight of labelling. The labels win --
    they are the only ground truth here, and they cost a person's time."""
    import app.retention as retention

    s = db()
    old = T0 - timedelta(days=400)
    for row in s.query(PlateRead).all():
        row.timestamp = old
    s.commit()
    review(s, 1, "B1234ABC")
    review(s, 4, "")
    s.commit()

    saved = retention.SessionLocal
    retention.SessionLocal = lambda: s
    try:
        result = retention.purge_old_data()
    finally:
        retention.SessionLocal = saved

    survivors = {r.id for r in s.query(PlateRead).all()}
    assert survivors == {1, 4}, survivors
    assert result["plate_reads"] == 2
    print(f"OK retention_keeps_reviewed_reads: purged {result['plate_reads']}, kept {survivors}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} plate-review tests passed.")
