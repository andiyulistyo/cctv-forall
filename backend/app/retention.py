"""Retention: delete count events and plate reads (plus their images) older
than ``settings.retention_days``. Runs periodically via APScheduler.

One exception, and it matters: a plate read somebody has **reviewed** is never
purged. Reviewing a read costs a person a few seconds of looking at a crop, and
the result -- what OCR said against what the plate really was -- is the only
ground truth this system has. Retention runs on a seven-day timer; without this
exemption a fortnight of labelling would simply evaporate, and it would do so
silently. Reviewed rows are a handful next to the traffic, and they are the
handful worth keeping.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import delete, select

from .config import settings
from .database import SessionLocal
from .models import CountEvent, FaceSighting, PlateRead

_scheduler: BackgroundScheduler | None = None


def purge_old_data() -> dict:
    # Naive UTC to match how timestamps are stored (see models._utcnow).
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        days=settings.retention_days
    )
    db = SessionLocal()
    removed_events = 0
    removed_plates = 0
    removed_sightings = 0
    try:
        # Delete plate images on disk first, then their rows.
        expired = (PlateRead.timestamp < cutoff, PlateRead.reviewed_at.is_(None))
        old_plates = db.scalars(select(PlateRead).where(*expired)).all()
        for pr in old_plates:
            # Both the crop and the full-frame evidence image.
            for rel in (pr.image_path, pr.frame_path):
                if not rel:
                    continue
                try:
                    (settings.data_dir / rel).unlink(missing_ok=True)
                except Exception:
                    pass
        removed_plates = db.execute(delete(PlateRead).where(*expired)).rowcount or 0

        # Face sightings (EnrolledFace is reference data and never purged).
        old_sightings = db.scalars(
            select(FaceSighting).where(FaceSighting.timestamp < cutoff)
        ).all()
        for fs in old_sightings:
            # Both the face crop and the full-frame evidence image.
            for rel in (fs.image_path, fs.frame_path):
                if not rel:
                    continue
                try:
                    (settings.data_dir / rel).unlink(missing_ok=True)
                except Exception:
                    pass
        removed_sightings = db.execute(
            delete(FaceSighting).where(FaceSighting.timestamp < cutoff)
        ).rowcount or 0

        removed_events = db.execute(
            delete(CountEvent).where(CountEvent.timestamp < cutoff)
        ).rowcount or 0
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
    return {
        "count_events": removed_events,
        "plate_reads": removed_plates,
        "face_sightings": removed_sightings,
    }


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
