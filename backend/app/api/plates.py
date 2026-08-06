"""License-plate reads: listing and image serving."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import get_current_user, validate_token
from ..config import settings
from ..database import get_db
from ..models import PlateRead
from ..schemas import PlateOut

router = APIRouter(prefix="/plates", tags=["plates"])


@router.get("", response_model=list[PlateOut], dependencies=[Depends(get_current_user)])
def list_plates(
    source: int | None = Query(None),
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = Query(None),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db),
) -> list[PlateOut]:
    stmt = select(PlateRead).order_by(PlateRead.timestamp.desc()).limit(limit)
    if source is not None:
        stmt = stmt.where(PlateRead.source_id == source)
    if from_ is not None:
        stmt = stmt.where(PlateRead.timestamp >= from_)
    if to is not None:
        stmt = stmt.where(PlateRead.timestamp <= to)

    rows = db.scalars(stmt).all()
    return [
        PlateOut(
            id=p.id,
            source_id=p.source_id,
            track_id=p.track_id,
            vehicle_class=p.vehicle_class,
            plate_text=p.plate_text,
            confidence=p.confidence,
            has_image=bool(p.image_path),
            timestamp=p.timestamp,
        )
        for p in rows
    ]


@router.get("/{plate_id}/image")
def plate_image(plate_id: int, token: str = Query(...), db: Session = Depends(get_db)):
    validate_token(token)
    p = db.get(PlateRead, plate_id)
    if not p or not p.image_path:
        raise HTTPException(404, "No image for this plate")
    fpath = settings.data_dir / p.image_path
    if not fpath.exists():
        raise HTTPException(404, "Image file missing")
    return FileResponse(str(fpath), media_type="image/jpeg")
