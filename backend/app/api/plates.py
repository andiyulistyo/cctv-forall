"""License-plate reads: listing and image serving."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth import get_current_user, validate_token
from ..config import settings
from ..database import get_db
from ..models import VEHICLE_CLASSES, PlateRead, Source
from ..schemas import PlateListResponse, PlateOut

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


@router.get("", response_model=PlateListResponse, dependencies=[Depends(get_current_user)])
def list_plates(
    source: int | None = Query(None),
    vehicle_class: str | None = Query(None, description="One of models.VEHICLE_CLASSES"),
    q: str | None = Query(None, description="Part of the plate text, case-insensitive"),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
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
        min_confidence=min_confidence, from_=from_, to=to,
        sort=sort, order=order, limit=limit, offset=offset,
    )


def query_plates(
    db: Session,
    *,
    source: int | None = None,
    vehicle_class: str | None = None,
    q: str | None = None,
    min_confidence: float | None = None,
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
    if q:
        # People type plates with the spaces and dashes the OCR never emits, so
        # search on whatever is left once those are gone.
        needle = "".join(ch for ch in q if ch.isalnum())
        if needle:
            filters.append(PlateRead.plate_text.ilike(f"%{needle}%"))

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
        plates=[
            PlateOut(
                id=p.id,
                source_id=p.source_id,
                track_id=p.track_id,
                vehicle_class=p.vehicle_class,
                plate_text=p.plate_text,
                confidence=p.confidence,
                has_image=bool(p.image_path),
                has_frame=bool(p.frame_path),
                timestamp=p.timestamp,
            )
            for p in rows
        ],
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
