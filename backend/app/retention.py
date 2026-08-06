"""Retention: delete count events and plate reads (plus their images) older
than ``settings.retention_days``. Runs periodically via APScheduler.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import delete, select

from .config import settings
from .database import SessionLocal
from .models import CountEvent, PlateRead

_scheduler: BackgroundScheduler | None = None


def purge_old_data() -> dict:
    # Naive UTC to match how timestamps are stored (see models._utcnow).
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        days=settings.retention_days
    )
    db = SessionLocal()
    removed_events = 0
    removed_plates = 0
    try:
        # Delete plate images on disk first, then their rows.
        old_plates = db.scalars(
            select(PlateRead).where(PlateRead.timestamp < cutoff)
        ).all()
        for pr in old_plates:
            if pr.image_path:
                fpath = settings.data_dir / pr.image_path
                try:
                    fpath.unlink(missing_ok=True)
                except Exception:
                    pass
        removed_plates = db.execute(
            delete(PlateRead).where(PlateRead.timestamp < cutoff)
        ).rowcount or 0

        removed_events = db.execute(
            delete(CountEvent).where(CountEvent.timestamp < cutoff)
        ).rowcount or 0
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
    return {"count_events": removed_events, "plate_reads": removed_plates}


def start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        purge_old_data,
        "interval",
        minutes=settings.retention_interval_minutes,
        id="retention",
        next_run_time=datetime.now(),  # run once at startup
    )
    _scheduler.start()


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
