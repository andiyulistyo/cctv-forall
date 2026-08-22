"""Tests for filtering and sorting the plate list.

These call query_plates directly against a throwaway in-memory database, so
there is no HTTP client, no auth and no data directory involved -- what is
under test is the query the filters build.

Run from the backend/ directory:  python -m tests.test_plate_filters
"""
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.plates import query_plates
from app.models import Base, PlateRead, Source

T0 = datetime(2026, 8, 22, 8, 0, 0)

# (source_id, class, plate, confidence, minutes after T0)
ROWS = [
    (1, "car", "B1234ABC", 0.95, 0),
    (1, "motorcycle", "B5678DEF", 0.40, 1),
    (1, "truck", "D9999XYZ", 0.72, 2),
    (2, "car", "B1111AAA", 0.88, 3),
    (2, "car", "F2222BBB", 0.51, 4),
    (2, "bus", "B3333CCC", 0.63, 5),
]
ALL_PLATES = [r[2] for r in ROWS]
FROM_SOURCE_2 = {"B1111AAA", "F2222BBB", "B3333CCC"}


def db():
    """A database holding two sources and ROWS worth of plate reads."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    # "Belakang" sorts before "Gerbang Utara" while their ids run the other
    # way, so a name sort and an id sort disagree -- the only way to prove
    # which one the endpoint actually does.
    s.add(Source(id=1, name="Gerbang Utara", type="rtsp", url="rtsp://a"))
    s.add(Source(id=2, name="Belakang", type="rtsp", url="rtsp://b"))
    for i, (src, cls, text, conf, mins) in enumerate(ROWS, start=1):
        s.add(
            PlateRead(
                id=i, source_id=src, track_id=i, vehicle_class=cls, plate_text=text,
                confidence=conf, timestamp=T0 + timedelta(minutes=mins),
            )
        )
    s.commit()
    return s


def plates(session=None, **kw) -> list[str]:
    """Plate texts the listing returns, in order."""
    return [p.plate_text for p in query_plates(session or db(), **kw).plates]


def test_default_is_newest_first():
    assert plates() == list(reversed(ALL_PLATES))
    print("OK default_is_newest_first")


def test_filter_by_class():
    s = db()
    assert plates(s, vehicle_class="car") == ["F2222BBB", "B1111AAA", "B1234ABC"]
    assert plates(s, vehicle_class="bus") == ["B3333CCC"]
    print("OK filter_by_class")


def test_filter_by_source_and_class_together():
    assert plates(source=2, vehicle_class="car") == ["F2222BBB", "B1111AAA"]
    print("OK filter_by_source_and_class_together")


def rejects(**kw) -> bool:
    try:
        query_plates(db(), **kw)
    except Exception as e:  # fastapi.HTTPException
        return getattr(e, "status_code", None) == 422
    return False


def test_a_class_that_cannot_have_a_plate_is_rejected():
    assert rejects(vehicle_class="bicycle")
    print("OK a_class_that_cannot_have_a_plate_is_rejected")


def test_a_cleared_class_dropdown_means_every_class():
    # "?vehicle_class=" is what a cleared dropdown sends -- not a class named "".
    assert plates(vehicle_class="") == list(reversed(ALL_PLATES))
    print("OK a_cleared_class_dropdown_means_every_class")


def test_an_unknown_sort_key_is_rejected():
    """Otherwise a typo would KeyError, or silently pick some other column."""
    assert rejects(sort="colour")
    print("OK an_unknown_sort_key_is_rejected")


def test_sort_by_confidence():
    s = db()
    assert plates(s, sort="confidence", order="desc")[0] == "B1234ABC"  # 0.95
    assert plates(s, sort="confidence", order="asc")[0] == "B5678DEF"   # 0.40
    print("OK sort_by_confidence")


def test_sort_by_plate_text():
    s = db()
    assert plates(s, sort="plate_text", order="asc")[0] == "B1111AAA"
    assert plates(s, sort="plate_text", order="desc")[0] == "F2222BBB"
    print("OK sort_by_plate_text")


def test_sort_by_source_uses_the_name_not_the_id():
    s = db()
    top3 = plates(s, sort="source", order="asc")[:3]
    assert set(top3) == FROM_SOURCE_2, top3
    bottom3 = plates(s, sort="source", order="desc")[:3]
    assert set(bottom3) == set(ALL_PLATES) - FROM_SOURCE_2, bottom3
    print("OK sort_by_source_uses_the_name_not_the_id")


def test_search_ignores_spaces_and_dashes():
    s = db()
    assert plates(s, q="b 1234-abc") == ["B1234ABC"]
    assert plates(s, q="2222") == ["F2222BBB"]
    assert plates(s, q="ZZZZ") == []
    print("OK search_ignores_spaces_and_dashes")


def test_a_search_of_only_punctuation_is_not_a_filter():
    """Stripping "- " down to nothing must mean no filter, not "match the
    empty string" -- which would quietly match every row anyway."""
    assert plates(q=" - ") == list(reversed(ALL_PLATES))
    print("OK a_search_of_only_punctuation_is_not_a_filter")


def test_min_confidence():
    assert plates(min_confidence=0.7) == ["B1111AAA", "D9999XYZ", "B1234ABC"]
    print("OK min_confidence")


def test_total_counts_every_match_not_just_the_page():
    r = query_plates(db(), limit=2)
    assert len(r.plates) == 2 and r.total == 6, (len(r.plates), r.total)
    assert query_plates(db(), vehicle_class="car", limit=1).total == 3
    print("OK total_counts_every_match_not_just_the_page")


def test_paging_never_repeats_or_drops_a_row():
    """Rows sharing a sort key must still land on exactly one page each."""
    s = db()
    seen = []
    for offset in (0, 2, 4):
        seen += plates(s, sort="vehicle_class", limit=2, offset=offset)
    assert sorted(seen) == sorted(ALL_PLATES), seen
    print("OK paging_never_repeats_or_drops_a_row")


def test_class_counts_ignore_the_class_filter():
    """Otherwise every count but the active one would read zero."""
    unfiltered = query_plates(db()).class_counts
    assert unfiltered == {"car": 3, "motorcycle": 1, "truck": 1, "bus": 1}, unfiltered
    picked = query_plates(db(), vehicle_class="bus")
    assert picked.class_counts == unfiltered, picked.class_counts
    assert picked.total == 1, picked.total
    print("OK class_counts_ignore_the_class_filter")


def test_class_counts_do_follow_the_other_filters():
    counts = query_plates(db(), source=2).class_counts
    assert counts == {"car": 2, "bus": 1}, counts
    print("OK class_counts_do_follow_the_other_filters")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} plate-filter tests passed.")
