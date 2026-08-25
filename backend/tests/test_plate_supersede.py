"""Tests for replacing a plate read's images with a better read's.

What this pins down: a vehicle is read several times as it approaches and only
the most confident read survives, so each improvement writes a fresh pair of
images and deletes the pair it replaced. The image a row *still points at* must
never be what gets deleted -- an operator opening that read gets a broken
thumbnail, and the evidence behind the read is gone for good.

That is not hypothetical. The read filename is stamped to the second, so two
improving reads of one vehicle inside the same second produce the *same* name:
the second write lands on top of the first, the row is pointed at a name it was
already pointing at, and the delete that follows takes the live file with it.

Run from the backend/ directory:  python -m tests.test_plate_supersede
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np

import app.detection.worker as worker
from app.config import settings


def _swap_data_dir(tmp: Path):
    """Point settings, the engine and the session factory at a scratch dir."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    settings.data_dir = tmp
    for d in (settings.plates_dir, settings.frames_dir):
        d.mkdir(parents=True, exist_ok=True)

    from app.database import Base
    import app.models  # noqa: F401  -- registers the tables on Base

    engine = create_engine(f"sqlite:///{tmp / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    worker.SessionLocal = sessionmaker(bind=engine, future=True)
    return engine


def _img(value: int) -> np.ndarray:
    return np.full((40, 120, 3), value, dtype=np.uint8)


def _paths_on_disk(row_id: int) -> tuple[str, str]:
    """The row's two image paths, asserted to exist."""
    from app.models import PlateRead

    db = worker.SessionLocal()
    try:
        row = db.get(PlateRead, row_id)
        assert row is not None, "the read vanished from the database"
        for rel in (row.image_path, row.frame_path):
            assert rel, f"read {row_id} has no image path"
            assert (settings.data_dir / rel).exists(), f"missing on disk: {rel}"
        return row.image_path, row.frame_path
    finally:
        db.close()


def test_better_read_keeps_its_own_images():
    """Two reads of one vehicle in the same second: the survivor keeps its files."""
    row_id = worker._persist_plate(
        7, 180, "car", "B1115UZJ", 0.42, _img(10),
        snapshot=_img(20), box=(1, 1, 30, 30),
    )
    assert row_id is not None
    _paths_on_disk(row_id)

    # The next frame reads the same plate more confidently. Same source, same
    # track, same text, same second -- so the same filename.
    again = worker._persist_plate(
        7, 180, "car", "B1115UZJ", 0.91, _img(11),
        snapshot=_img(21), box=(1, 1, 30, 30),
        row_id=row_id,
    )
    assert again == row_id, "a correction must stay on the same row"
    _paths_on_disk(row_id)


def test_superseded_images_are_cleaned_up():
    """A genuinely replaced pair is still deleted -- no orphan files."""
    row_id = worker._persist_plate(8, 5, "car", "B1111AAA", 0.30, _img(10), snapshot=_img(20))
    first = _paths_on_disk(row_id)

    # A different text gives a different filename, so this is a real supersede.
    worker._persist_plate(
        8, 5, "car", "B2222BBB", 0.80, _img(11), snapshot=_img(21), row_id=row_id
    )
    second = _paths_on_disk(row_id)

    assert second != first, "the better read should have written new files"
    for rel in first:
        assert not (settings.data_dir / rel).exists(), f"orphan left behind: {rel}"


def test_a_capture_row_is_filled_in_by_a_later_read():
    """A car recorded unread, then read: one row, not two.

    The whole reason a read class is captured at all -- a car whose plate the
    night defeats still has to be a row -- rests on the plate landing on that
    same row when one is finally made out. The follow-up read carries no
    snapshot of its own, so the evidence frame the capture wrote has to survive
    it.
    """
    from app.models import PlateRead

    row_id = worker._persist_plate(
        9, 42, "car", "", 0.0, _img(10), snapshot=_img(20), box=(1, 1, 30, 30),
    )
    assert row_id is not None
    _, captured_frame = _paths_on_disk(row_id)

    same = worker._persist_plate(
        9, 42, "car", "B1234XYZ", 0.70, _img(11), snapshot=None,
        box=(1, 1, 30, 30), row_id=row_id,
    )
    assert same == row_id, "the read must fill the capture's row, not add one"

    db = worker.SessionLocal()
    try:
        rows = db.query(PlateRead).filter(PlateRead.source_id == 9).all()
        assert len(rows) == 1, f"{len(rows)} rows for one passage"
        assert rows[0].plate_text == "B1234XYZ", rows[0].plate_text
        assert rows[0].frame_path == captured_frame, "the evidence frame was dropped"
    finally:
        db.close()
    _paths_on_disk(row_id)


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    engine = _swap_data_dir(tmp)
    try:
        failures = 0
        for name, fn in sorted(globals().items()):
            if not name.startswith("test_"):
                continue
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
        print("ok" if not failures else f"{failures} failure(s)")
    finally:
        # Windows will not remove a sqlite file the engine still holds open.
        engine.dispose()
        shutil.rmtree(tmp, ignore_errors=True)
    raise SystemExit(1 if failures else 0)
