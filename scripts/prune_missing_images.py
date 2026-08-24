#!/usr/bin/env python
"""Clear plate and face image paths whose file is no longer on disk.

A row that points at a missing file renders as a broken thumbnail in the
dashboard: the listing says there is an image, the request for it 404s, and
the operator is left with a torn-page icon and no way to tell whether the
capture failed or the file was lost. A row with *no* path renders as a plain
"—", which is at least honest about what is there.

The images themselves are gone and nothing here brings them back. This only
stops the dashboard claiming otherwise.

Dry run by default -- it reports what it would clear and changes nothing:

    cd backend && .venv/Scripts/python ../scripts/prune_missing_images.py
    .venv/Scripts/python ../scripts/prune_missing_images.py --apply
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import FaceSighting, PlateRead  # noqa: E402


def prune(apply: bool) -> int:
    db = SessionLocal()
    cleared = Counter()
    try:
        for model, label in ((PlateRead, "plate read"), (FaceSighting, "face sighting")):
            for row in db.scalars(select(model)).all():
                for field in ("image_path", "frame_path"):
                    rel = getattr(row, field)
                    if not rel or (settings.data_dir / rel).exists():
                        continue
                    print(f"{label} {row.id}: {field} -> {rel}")
                    cleared[label] += 1
                    if apply:
                        setattr(row, field, None)
        if apply:
            db.commit()
    finally:
        db.close()

    total = sum(cleared.values())
    if not total:
        print("Nothing to clear: every path on record has its file.")
    else:
        for label, n in sorted(cleared.items()):
            print(f"{n} dangling path(s) on {label} rows")
        print("Cleared." if apply else "Dry run -- pass --apply to clear these.")
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the changes")
    prune(ap.parse_args().apply)
