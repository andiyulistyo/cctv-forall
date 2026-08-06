"""SQLAlchemy ORM models."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _utcnow() -> datetime:
    # Naive UTC: SQLite's DATETIME columns are timezone-naive, so storing
    # tz-aware values would make range comparisons (e.g. retention) unreliable.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))


# Valid source input types.
SOURCE_TYPES = ("youtube", "rtsp", "rtmp", "hls", "http", "file")

# Supported detection classes (COCO name -> friendly). Keys are used everywhere.
DETECTION_CLASSES = ("person", "car", "motorcycle", "truck", "bus")
# Classes that are vehicles (eligible for ANPR / plate reading).
VEHICLE_CLASSES = ("car", "motorcycle", "truck", "bus")


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    type: Mapped[str] = mapped_column(String(20))
    url: Mapped[str] = mapped_column(Text)

    # List[str] of enabled class names, e.g. ["car", "truck", "motorcycle"].
    enabled_classes: Mapped[list] = mapped_column(JSON, default=list)

    # Counting line, normalized coordinates 0..1: {"a": [x, y], "b": [x, y]}.
    # None until the user draws a line.
    line: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Human labels for the two crossing directions, e.g.
    # {"in": "masuk", "out": "keluar"}.
    direction_labels: Mapped[dict] = mapped_column(
        JSON, default=lambda: {"in": "in", "out": "out"}
    )

    alpr_enabled: Mapped[bool] = mapped_column(Integer, default=1)

    # Runtime status: stopped | starting | running | error
    status: Mapped[str] = mapped_column(String(20), default="stopped")
    status_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    count_events: Mapped[list["CountEvent"]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )
    plate_reads: Mapped[list["PlateRead"]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


class CountEvent(Base):
    """One line-crossing event for a single tracked object."""

    __tablename__ = "count_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), index=True
    )
    class_name: Mapped[str] = mapped_column(String(30), index=True)
    direction: Mapped[str] = mapped_column(String(10))  # "in" | "out"
    track_id: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    source: Mapped[Source] = relationship(back_populates="count_events")


class PlateRead(Base):
    """A license-plate OCR result for a detected vehicle."""

    __tablename__ = "plate_reads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), index=True
    )
    track_id: Mapped[int] = mapped_column(Integer)
    vehicle_class: Mapped[str] = mapped_column(String(30))
    plate_text: Mapped[str] = mapped_column(String(30), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    source: Mapped[Source] = relationship(back_populates="plate_reads")
