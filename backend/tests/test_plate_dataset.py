"""Tests for app.plate_dataset -- the scoring, the split, and the export.

This module feeds two consumers that must never disagree: the Akurasi OCR page
and scripts/export_plate_dataset.py. What it computes is also the only evidence
anyone will have when deciding whether to replace the recogniser, so a wrong
number here is not a cosmetic bug -- it is a wrong decision with a chart behind
it.

Two failures in particular are silent by construction and are pinned hardest. A
broken edit distance reports a character error rate that is merely a number. A
broken split reports an accuracy that is *flattering*: if one plate can land in
train and in eval, the eval score is partly a memory test.

Run from the backend/ directory:  python -m tests.test_plate_dataset
"""
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import plate_dataset as export
from app.config import settings
from app.models import Base, PlateRead, Source

T0 = datetime(2026, 8, 22, 8, 0, 0)

# (id, source, what OCR read, what it really is, reviewed?)
# Two reads of B1234ABC on purpose: one plate, two rows, and the split must
# keep them together.
ROWS = [
    (1, 1, "B1234ABC", "B1234ABC", True),
    (2, 1, "B5678DEF", "B5678DEE", True),    # one character wrong
    (3, 1, "B1234ABC", "B1234ABC", True),    # same plate as row 1
    (4, 2, "", "D9999XYZ", True),            # read nothing at all
    (5, 2, "F2222BBB", "", True),            # reviewed as illegible
    (6, 2, "B3333CCC", None, False),         # never reviewed
]


def db_with_crops(tmp: Path):
    """A scratch database whose crops actually exist on disk."""
    settings.data_dir = tmp
    (tmp / "plates").mkdir(parents=True, exist_ok=True)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(Source(id=1, name="Gerbang Utara", type="rtsp", url="rtsp://a"))
    s.add(Source(id=2, name="Belakang", type="rtsp", url="rtsp://b"))
    for i, (rid, src, pred, truth, reviewed) in enumerate(ROWS):
        rel = f"plates/{rid}.jpg"
        (tmp / rel).write_bytes(b"\xff\xd8" + bytes(200) + b"\xff\xd9")
        s.add(PlateRead(
            id=rid, source_id=src, track_id=rid, vehicle_class="car",
            plate_text=pred, confidence=0.6, image_path=rel,
            corrected_text=truth if reviewed else None,
            reviewed_at=T0 if reviewed else None,
            reviewed_by="admin" if reviewed else None,
            timestamp=T0 + timedelta(minutes=i),
        ))
    s.commit()
    return s


# --- character error rate ---------------------------------------------------

def test_edit_distance_counts_what_it_says_it_counts():
    assert export.levenshtein("B1234XYZ", "B1234XYZ") == 0
    assert export.levenshtein("B1234XY2", "B1234XYZ") == 1   # one substitution
    assert export.levenshtein("1234XYZ", "B1234XYZ") == 1    # one insertion
    assert export.levenshtein("", "B1234XYZ") == 8           # read nothing at all
    print("OK edit_distance_counts_what_it_says_it_counts")


# --- the split --------------------------------------------------------------

def test_a_plate_always_lands_in_the_same_split():
    """Several reads of one car share a label. If they could be split apart,
    the eval set would contain plates the model was trained on and every score
    after that would be too good."""
    for plate in ("B1234XYZ", "AD12AB", "F2222BBB"):
        picks = {export.split_for(plate, 0.2) for _ in range(50)}
        assert len(picks) == 1, f"{plate} landed in {picks}"
    print("OK a_plate_always_lands_in_the_same_split")


def test_the_split_is_stable_across_runs_so_two_scores_are_comparable():
    before = {p: export.split_for(p, 0.2) for p in ("B1", "B2", "B3", "B4", "B5")}
    after = {p: export.split_for(p, 0.2) for p in ("B5", "B4", "B3", "B2", "B1")}
    assert before == after
    print("OK the_split_is_stable_across_runs")


def test_the_holdout_share_is_roughly_what_was_asked_for():
    plates = [f"B{n:04d}XYZ" for n in range(2000)]
    got = Counter(export.split_for(p, 0.2) for p in plates)
    share = got["eval"] / len(plates)
    assert 0.17 < share < 0.23, f"asked for 0.2, got {share:.3f}"
    print(f"OK the_holdout_share_is_roughly_what_was_asked_for: {share:.3f}")


def test_asking_for_no_holdout_gives_none():
    assert all(export.split_for(f"B{n}", 0.0) == "train" for n in range(200))
    print("OK asking_for_no_holdout_gives_none")


# --- progress ---------------------------------------------------------------

def test_progress_splits_the_reviews_the_way_the_page_reports_them():
    with tempfile.TemporaryDirectory() as tmp:
        p = export.review_progress(db_with_crops(Path(tmp)))
    assert p["total"] == 6
    assert p["reviewed"] == 5
    assert p["pending"] == 1
    assert p["correct"] == 2          # rows 1 and 3
    assert p["wrong"] == 2            # rows 2 and 4
    assert p["illegible"] == 1        # row 5
    # The three outcomes have to add up to what was reviewed, or the page is
    # quietly losing rows into a category nobody shows.
    assert p["correct"] + p["wrong"] + p["illegible"] == p["reviewed"]
    print("OK progress_splits_the_reviews_the_way_the_page_reports_them")


def test_a_read_with_no_text_at_all_counts_as_wrong_not_as_correct():
    """Row 4: OCR returned "", the plate is D9999XYZ. Both are "not equal", but
    an empty prediction is the failure this system most wants to see counted."""
    with tempfile.TemporaryDirectory() as tmp:
        s = db_with_crops(Path(tmp))
        p = export.review_progress(s)
        items, _ = export.collect(s)
    assert p["correct"] == 2, "an empty read was scored as a correct one"
    assert export.score(items).blank == 1
    print("OK a_read_with_no_text_at_all_counts_as_wrong_not_as_correct")


# --- collect ----------------------------------------------------------------

def test_illegible_and_unreviewed_reads_stay_out_of_the_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        items, skipped = export.collect(db_with_crops(Path(tmp)))
    assert {i.id for i in items} == {1, 2, 3, 4}
    assert skipped["reviewed as illegible"] == 1
    print("OK illegible_and_unreviewed_reads_stay_out_of_the_dataset")


def test_a_crop_that_is_no_longer_on_disk_is_reported_not_silently_dropped():
    """Retention and a hand-tidied folder both do this. "412 reads scored"
    means something different when 300 more went missing."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        s = db_with_crops(root)
        (root / "plates" / "1.jpg").unlink()
        items, skipped = export.collect(s)
    assert 1 not in {i.id for i in items}
    assert skipped["crop missing from disk"] == 1
    print("OK a_crop_that_is_no_longer_on_disk_is_reported")


# --- scoring ----------------------------------------------------------------

def test_the_score_is_the_one_the_page_and_the_cli_both_read():
    with tempfile.TemporaryDirectory() as tmp:
        items, _ = export.collect(db_with_crops(Path(tmp)))
        m = export.score(items)
    assert m.reads == 4
    assert m.exact == 2                      # rows 1 and 3
    assert abs(m.exact_rate - 0.5) < 1e-9
    assert m.blank == 1                      # row 4
    # 8 chars per label x 4 = 32; row 2 costs one substitution, row 4 costs
    # eight insertions.
    assert m.chars == 32
    assert m.errors == 9
    assert abs(m.cer - 9 / 32) < 1e-9
    print(f"OK the_score_is_the_one_the_page_and_the_cli_both_read: {m.as_dict()['cer']}")


def test_the_confusion_table_only_pairs_characters_it_can_actually_pair():
    """Row 2 is F->E at the same length, which is a real pair. Row 4 read
    nothing at all, and inventing eight pairs out of that would poison the
    table that alpr._CONFUSIONS is meant to be built from."""
    with tempfile.TemporaryDirectory() as tmp:
        items, _ = export.collect(db_with_crops(Path(tmp)))
        m = export.score(items)
    assert m.confusions == [("F->E", 1)], m.confusions
    print("OK the_confusion_table_only_pairs_characters_it_can_actually_pair")


def test_per_source_scores_are_kept_apart():
    with tempfile.TemporaryDirectory() as tmp:
        items, _ = export.collect(db_with_crops(Path(tmp)))
        by_source = export.score_by_source(items)
    assert set(by_source) == {1, 2}
    assert by_source[1].reads == 3 and by_source[1].exact == 2
    assert by_source[2].reads == 1 and by_source[2].exact == 0
    print("OK per_source_scores_are_kept_apart")


def test_scoring_nothing_is_zeroes_and_not_a_crash():
    m = export.score([])
    assert (m.reads, m.exact_rate, m.cer, m.blank_rate) == (0, 0.0, 0.0, 0.0)
    print("OK scoring_nothing_is_zeroes_and_not_a_crash")


# --- the archive ------------------------------------------------------------

def test_the_zip_holds_every_crop_its_labels_and_a_note_on_what_it_is():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        items, _ = export.collect(db_with_crops(root))
        dest = root / "out" / "plate-dataset.zip"
        counts = export.write_archive(items, dest, 0.2)

        with zipfile.ZipFile(dest) as zf:
            names = zf.namelist()
            csv_text = zf.read("labels.csv").decode("utf-8")
            jsonl = zf.read("labels.jsonl").decode("utf-8").splitlines()
            assert zf.read("README.txt")

    crops = [n for n in names if n.endswith(".jpg")]
    assert len(crops) == 4 == counts["train"] + counts["eval"]
    assert len(jsonl) == 4
    assert csv_text.splitlines()[0] == "path,label"
    # Every crop is named by row id, so two reads of one plate cannot collide
    # and quietly shrink the dataset.
    assert len(set(crops)) == len(crops)
    print(f"OK the_zip_holds_every_crop_its_labels_and_a_note: {sorted(crops)}")


def test_two_reads_of_one_plate_never_straddle_the_split():
    """Rows 1 and 3 are the same plate. If they land either side of the split,
    the eval score is measuring recall of a plate that was trained on."""
    import json

    splits = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        items, _ = export.collect(db_with_crops(root))
        out = root / "ds"
        export.write_dataset(items, out, 0.2)
        for line in (out / "labels.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            splits.setdefault(r["label"], set()).add(r["split"])
    leaky = {k: v for k, v in splits.items() if len(v) > 1}
    assert not leaky, f"one plate landed in two splits: {leaky}"
    assert len(splits["B1234ABC"]) == 1
    print(f"OK two_reads_of_one_plate_never_straddle_the_split ({len(splits)} plates)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} plate-dataset tests passed.")
