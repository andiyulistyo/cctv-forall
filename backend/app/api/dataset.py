"""OCR accuracy: what the reviews say about the reader, and the export.

This is the second half of the review loop. Reviewing a read (see
``api/plates.py``) produces ground truth one row at a time; this turns the pile
of it into the two things that ground truth is *for* -- a number that says how
good the reader actually is on these cameras, and a dataset that can train or
judge a replacement.

Both used to be a command line away, which meant that in practice neither
happened: the person who reviews the plates is not the person with a shell on
the box. The arithmetic itself lives in ``app.plate_dataset`` and is shared
with ``scripts/export_plate_dataset.py``, so the page and the CLI can never
quote different accuracies for the same rows.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from ..auth import get_current_user, validate_token
from ..database import get_db
from ..models import Source
from ..plate_dataset import collect, review_progress, score, score_by_source, write_archive

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("", dependencies=[Depends(get_current_user)])
def dataset_summary(db: Session = Depends(get_db)) -> dict:
    """Review coverage, the reader's score, and the score per camera.

    ``metrics`` is null until something has been reviewed -- deliberately null
    rather than a row of zeroes, which would read as "the reader gets nothing
    right" when it means "nobody has checked".
    """
    progress = review_progress(db)
    items, skipped = collect(db)
    names = {s.id: s.name for s in db.query(Source).all()}

    per_source = [
        {
            "source_id": sid,
            "name": names.get(sid, f"#{sid}"),
            **m.as_dict(),
        }
        for sid, m in sorted(
            score_by_source(items).items(),
            # Worst first: the point of this table is to find the camera worth
            # walking over to, and that one is never at the bottom of a list
            # sorted by id.
            key=lambda kv: kv[1].exact_rate,
        )
    ]

    return {
        "progress": progress,
        "metrics": score(items).as_dict() if items else None,
        "per_source": per_source,
        "skipped": [{"reason": r, "count": n} for r, n in skipped.most_common()],
    }


@router.get("/export")
def export_dataset(
    token: str = Query(..., description="Auth token; the browser downloads this directly"),
    eval_share: float = Query(0.2, ge=0.0, le=0.9),
    db: Session = Depends(get_db),
):
    """The crops and their labels, as one zip.

    The token rides in the query string because this is fetched by navigating
    to it rather than with fetch(), which is what gives the browser its own
    download UI and progress bar -- the same trade the image endpoints already
    make.

    The archive is built into a temp file and deleted once it has been sent.
    Holding it in memory would work today and stop working on the first fully
    reviewed camera-year.
    """
    validate_token(token)
    items, _ = collect(db)
    if not items:
        raise HTTPException(
            404,
            "Nothing to export yet -- no plate read has been reviewed. "
            "Open Plat Nomor, filter to 'Belum ditinjau', and correct a few.",
        )

    tmp = Path(tempfile.mkdtemp(prefix="plate-dataset-")) / "plate-dataset.zip"
    write_archive(items, tmp, eval_share)
    return FileResponse(
        str(tmp),
        media_type="application/zip",
        filename="plate-dataset.zip",
        # Runs after the response is finished, whether or not it finished well.
        background=BackgroundTask(lambda: _cleanup(tmp)),
    )


def _cleanup(path: Path) -> None:
    import shutil

    shutil.rmtree(path.parent, ignore_errors=True)
