"""License-plate reads: listing and image serving."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..auth import get_current_user, validate_token
from ..config import settings
from ..database import get_db
from ..detection.alpr import normalize_plate
from ..models import VEHICLE_CLASSES, PlateRead, Source
from ..schemas import PlateListResponse, PlateOut, PlateReviewIn

router = APIRouter(prefix="/plates", tags=["plates"])

# What `sort` may order by. "source" means the operator-visible source *name*,
# not the id -- sorting by a number nobody sees is not a sort anyone asked for.
SORT_COLUMNS = {
    "timestamp": PlateRead.timestamp,
    "confidence": PlateRead.confidence,
    "plate_text": PlateRead.plate_text,
    "vehicle_class": PlateRead.vehicle_class,
    "source": Source.name,
}
SortKey = Literal["timestamp", "confidence", "plate_text", "vehicle_class", "source"]

# Where a read stands with a human reviewer.
#
#   pending   nobody has looked at it
#   reviewed  somebody has, whatever they concluded
#   correct   they looked and OCR had it right
#   wrong     they looked and OCR had it wrong -- the rows worth studying
#   illegible they looked and the plate cannot be read from the crop at all
#
# "wrong" is the one that earns its keep. It is the error set: every read the
# recogniser got wrong, next to the picture it got it wrong from.
ReviewFilter = Literal["pending", "reviewed", "correct", "wrong", "illegible"]

_REVIEW_FILTERS = {
    "pending": lambda: (PlateRead.reviewed_at.is_(None),),
    "reviewed": lambda: (PlateRead.reviewed_at.is_not(None),),
    "correct": lambda: (
        PlateRead.reviewed_at.is_not(None),
        PlateRead.corrected_text != "",
        PlateRead.corrected_text == PlateRead.plate_text,
    ),
    "wrong": lambda: (
        PlateRead.reviewed_at.is_not(None),
        PlateRead.corrected_text != "",
        PlateRead.corrected_text != PlateRead.plate_text,
    ),
    "illegible": lambda: (
        PlateRead.reviewed_at.is_not(None),
        PlateRead.corrected_text == "",
    ),
}


def _plate_out(p: PlateRead) -> PlateOut:
    return PlateOut(
        id=p.id,
        source_id=p.source_id,
        track_id=p.track_id,
        vehicle_class=p.vehicle_class,
        plate_text=p.plate_text,
        confidence=p.confidence,
        has_image=bool(p.image_path),
        has_frame=bool(p.frame_path),
        corrected_text=p.corrected_text,
        reviewed_at=p.reviewed_at,
        reviewed_by=p.reviewed_by,
        timestamp=p.timestamp,
    )


@router.get("", response_model=PlateListResponse, dependencies=[Depends(get_current_user)])
def list_plates(
    source: int | None = Query(None),
    vehicle_class: str | None = Query(None, description="One of models.VEHICLE_CLASSES"),
    q: str | None = Query(None, description="Part of the plate text, case-insensitive"),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    review: ReviewFilter | None = Query(None, description="Where the read stands with a reviewer"),
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = Query(None),
    sort: SortKey = Query("timestamp"),
    order: Literal["asc", "desc"] = Query("desc"),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> PlateListResponse:
    return query_plates(
        db, source=source, vehicle_class=vehicle_class, q=q,
        min_confidence=min_confidence, review=review, from_=from_, to=to,
        sort=sort, order=order, limit=limit, offset=offset,
    )


def query_plates(
    db: Session,
    *,
    source: int | None = None,
    vehicle_class: str | None = None,
    q: str | None = None,
    min_confidence: float | None = None,
    review: str | None = None,
    from_: datetime | None = None,
    to: datetime | None = None,
    sort: SortKey = "timestamp",
    order: Literal["asc", "desc"] = "desc",
    limit: int = 100,
    offset: int = 0,
) -> PlateListResponse:
    """The listing itself, kept out of the route so it can be exercised with
    plain arguments -- a Query(...) default is a marker object until FastAPI
    has parsed a real request, which makes the route unusable as a function."""
    # "?vehicle_class=" is how a cleared dropdown looks, and it means every
    # class -- not a class named "".
    vehicle_class = vehicle_class or None
    if vehicle_class is not None and vehicle_class not in VEHICLE_CLASSES:
        raise HTTPException(422, f"Unknown vehicle class: {vehicle_class}")
    if sort not in SORT_COLUMNS:
        raise HTTPException(422, f"Cannot sort by: {sort}")

    filters = []
    if source is not None:
        filters.append(PlateRead.source_id == source)
    if from_ is not None:
        filters.append(PlateRead.timestamp >= from_)
    if to is not None:
        filters.append(PlateRead.timestamp <= to)
    if min_confidence is not None:
        filters.append(PlateRead.confidence >= min_confidence)
    if review:
        if review not in _REVIEW_FILTERS:
            raise HTTPException(422, f"Unknown review filter: {review}")
        filters.extend(_REVIEW_FILTERS[review]())
    if q:
        # People type plates with the spaces and dashes the OCR never emits, so
        # search on whatever is left once those are gone.
        needle = "".join(ch for ch in q if ch.isalnum())
        if needle:
            # Either text: somebody searching for a plate wants the read whose
            # *correction* is that plate just as much as the one OCR happened
            # to spell right, and after a review the correction is the only
            # place the real number is written down.
            filters.append(
                or_(
                    PlateRead.plate_text.ilike(f"%{needle}%"),
                    PlateRead.corrected_text.ilike(f"%{needle}%"),
                )
            )

    # Per-class totals, so the class filter can say what picking it would get
    # you. They deliberately ignore the class filter itself -- applied, every
    # class but the active one would read zero and the counts would be useless.
    class_counts = dict(
        db.execute(
            select(PlateRead.vehicle_class, func.count())
            .where(*filters)
            .group_by(PlateRead.vehicle_class)
        ).all()
    )

    if vehicle_class is not None:
        filters.append(PlateRead.vehicle_class == vehicle_class)

    # The caller needs the unpaged total to know how many pages there are, and
    # the class counts already add up to exactly that -- no second COUNT query.
    total = class_counts.get(vehicle_class, 0) if vehicle_class else sum(class_counts.values())

    stmt = select(PlateRead).where(*filters)
    if sort == "source":
        stmt = stmt.join(Source, PlateRead.source_id == Source.id)

    col = SORT_COLUMNS[sort]
    # id breaks ties: reads routinely share a timestamp, a class or a source,
    # and without a stable second key the same row could show up on two pages
    # (or on none).
    stmt = (
        stmt.order_by(col.asc() if order == "asc" else col.desc(), PlateRead.id.desc())
        .limit(limit)
        .offset(offset)
    )

    rows = db.scalars(stmt).all()
    return PlateListResponse(
        plates=[_plate_out(p) for p in rows],
        total=total,
        class_counts=class_counts,
    )


def _serve(rel_path: str | None, missing: str) -> FileResponse:
    if not rel_path:
        raise HTTPException(404, missing)
    fpath = settings.data_dir / rel_path
    if not fpath.exists():
        raise HTTPException(404, "Image file missing")
    return FileResponse(str(fpath), media_type="image/jpeg")


@router.get("/{plate_id}/image")
def plate_image(plate_id: int, token: str = Query(...), db: Session = Depends(get_db)):
    """The cropped plate -- what the OCR actually read."""
    validate_token(token)
    p = db.get(PlateRead, plate_id)
    if not p:
        raise HTTPException(404, "No image for this plate")
    return _serve(p.image_path, "No image for this plate")


@router.get("/{plate_id}/frame")
def plate_frame(plate_id: int, token: str = Query(...), db: Session = Depends(get_db)):
    """The full frame at the moment of the read, with the vehicle boxed."""
    validate_token(token)
    p = db.get(PlateRead, plate_id)
    if not p:
        raise HTTPException(404, "No frame for this plate")
    return _serve(p.frame_path, "No frame for this plate")


@router.put("/{plate_id}/review", response_model=PlateOut)
def review_plate(
    plate_id: int,
    body: PlateReviewIn,
    db: Session = Depends(get_db),
    user: str = Depends(get_current_user),
) -> PlateOut:
    """Record what a person says this plate actually is.

    ``plate_text`` is normalised the same way an OCR result is -- upper case,
    nothing but letters and digits -- so that a human typing "B 6084 TXB" and a
    reader emitting "B6084TXB" produce the same label. Comparing a prediction
    against a label that differs only in spacing would count every correct read
    as an error.

    Empty means illegible, and is stored as such. It is not a rejected input:
    a crop nobody can read is a fact about the crop, and recording it is what
    keeps that crop out of a training set rather than in it under a guess.

    Idempotent, and a review can be revised -- the row keeps the latest answer
    and who gave it. The OCR text itself is never touched.
    """
    row = db.get(PlateRead, plate_id)
    if row is None:
        raise HTTPException(404, "No such plate read")

    text = normalize_plate(body.plate_text)
    if body.plate_text.strip() and not text:
        # They typed something, and nothing survived normalisation -- so what
        # they typed had no letters or digits in it at all. Saying so beats
        # silently filing it as "illegible", which is a different claim.
        raise HTTPException(422, "A plate needs at least one letter or digit")

    row.corrected_text = text
    row.reviewed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    row.reviewed_by = user
    db.commit()
    db.refresh(row)
    return _plate_out(row)


@router.delete("/{plate_id}/review", response_model=PlateOut)
def unreview_plate(
    plate_id: int,
    db: Session = Depends(get_db),
    user: str = Depends(get_current_user),
) -> PlateOut:
    """Undo a review, putting the read back in the pending queue.

    For the misclick and for the second thought, both of which happen when a
    person is working through a few hundred crops in one sitting.
    """
    row = db.get(PlateRead, plate_id)
    if row is None:
        raise HTTPException(404, "No such plate read")
    row.corrected_text = None
    row.reviewed_at = None
    row.reviewed_by = None
    db.commit()
    db.refresh(row)
    return _plate_out(row)
