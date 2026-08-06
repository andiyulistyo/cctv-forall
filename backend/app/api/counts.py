"""Aggregated counting statistics."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..detection.manager import get_manager
from ..models import CountEvent
from ..schemas import CountBucket, CountsResponse

router = APIRouter(prefix="/counts", tags=["counts"], dependencies=[Depends(get_current_user)])


@router.get("", response_model=CountsResponse)
def get_counts(
    source: int | None = Query(None, description="Filter by source id"),
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = Query(None),
    db: Session = Depends(get_db),
) -> CountsResponse:
    stmt = select(
        CountEvent.class_name,
        CountEvent.direction,
        func.count().label("n"),
    ).group_by(CountEvent.class_name, CountEvent.direction)

    if source is not None:
        stmt = stmt.where(CountEvent.source_id == source)
    if from_ is not None:
        stmt = stmt.where(CountEvent.timestamp >= from_)
    if to is not None:
        stmt = stmt.where(CountEvent.timestamp <= to)

    rows = db.execute(stmt).all()
    buckets = [CountBucket(class_name=r[0], direction=r[1], count=r[2]) for r in rows]

    live = None
    if source is not None:
        live = get_manager().get_counts(source)

    return CountsResponse(source_id=source, buckets=buckets, live=live)
