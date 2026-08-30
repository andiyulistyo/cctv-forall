"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_serializer

from .models import DETECTION_CLASSES, SOURCE_TYPES


def _as_utc_iso(v: datetime | None) -> str | None:
    """Serialize a naive-UTC datetime as an explicit UTC ISO string."""
    if v is None:
        return None
    return v.isoformat() + "Z"


# --- Auth ---
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# --- Line ---
class Line(BaseModel):
    a: list[float] = Field(..., min_length=2, max_length=2)  # [x, y] normalized 0..1
    b: list[float] = Field(..., min_length=2, max_length=2)


class DirectionLabels(BaseModel):
    in_: str = Field("in", alias="in")
    out: str = "out"

    model_config = {"populate_by_name": True}


class LineUpdate(BaseModel):
    line: Line
    direction_labels: DirectionLabels | None = None


# --- Plate-reading zone ---
class Zone(BaseModel):
    """Two opposite corners of a rectangle, normalized 0..1.

    Which corner is which is not fixed: the drawing UI reports wherever the
    drag started and ended, and the worker sorts them.
    """

    a: list[float] = Field(..., min_length=2, max_length=2)
    b: list[float] = Field(..., min_length=2, max_length=2)


class ZoneUpdate(BaseModel):
    # None clears the zone, restoring "read anywhere in the frame".
    zone: Zone | None = None


# --- Sources ---
class SourceBase(BaseModel):
    name: str
    type: str
    url: str
    enabled_classes: list[str] = Field(default_factory=lambda: ["car", "truck", "motorcycle"])
    alpr_enabled: bool = True
    face_enabled: bool = False
    # Resume this source on app start (see Source.auto_start).
    auto_start: bool = False

    def validate_semantics(self) -> None:
        if self.type not in SOURCE_TYPES:
            raise ValueError(f"type must be one of {SOURCE_TYPES}")
        bad = [c for c in self.enabled_classes if c not in DETECTION_CLASSES]
        if bad:
            raise ValueError(f"unknown classes: {bad}; allowed {DETECTION_CLASSES}")


class SourceCreate(SourceBase):
    pass


class SourceUpdate(BaseModel):
    name: str | None = None
    url: str | None = None
    enabled_classes: list[str] | None = None
    alpr_enabled: bool | None = None
    face_enabled: bool | None = None
    auto_start: bool | None = None


class SourceOut(SourceBase):
    id: int
    line: dict | None = None
    alpr_zone: dict | None = None
    direction_labels: dict
    status: str
    status_message: str | None = None
    created_at: datetime
    # Live throughput of the worker (capture/detect/publish fps), None unless
    # the source is running. Diagnostic only — see _Throughput in worker.py.
    stats: dict | None = None

    model_config = {"from_attributes": True}

    @field_serializer("created_at")
    def _ser_created_at(self, v: datetime) -> str | None:
        return _as_utc_iso(v)


class SourceListResponse(BaseModel):
    sources: list[SourceOut]
    total: int
    active: int  # number of running sources


# --- Counts ---
class CountBucket(BaseModel):
    class_name: str
    direction: str
    count: int


class CountsResponse(BaseModel):
    source_id: int | None = None
    buckets: list[CountBucket]
    # live per-class counters straight from the running worker (if any)
    live: dict | None = None


# --- Plates ---
class PlateOut(BaseModel):
    id: int
    source_id: int
    track_id: int
    vehicle_class: str
    # What OCR read. Never overwritten by a review -- see models.PlateRead.
    plate_text: str
    confidence: float
    has_image: bool
    has_frame: bool
    # What a person said it really is. None = nobody has looked at it yet;
    # "" = somebody looked and it is not legible.
    corrected_text: str | None = None
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None
    timestamp: datetime

    model_config = {"from_attributes": True}

    @field_serializer("timestamp")
    def _ser_timestamp(self, v: datetime) -> str | None:
        return _as_utc_iso(v)

    @field_serializer("reviewed_at")
    def _ser_reviewed_at(self, v: datetime | None) -> str | None:
        return _as_utc_iso(v) if v is not None else None


class PlateReviewIn(BaseModel):
    """One person's answer to "what does this plate actually say?".

    The empty string is a legitimate answer and means "I looked, and it cannot
    be read" -- which is worth recording, because it is what keeps an
    unreadable crop out of a training set instead of into it under a guess.
    """

    plate_text: str = Field("", max_length=30)


class PlateListResponse(BaseModel):
    plates: list[PlateOut]
    # total matching the filters, ignoring limit/offset -- the page count
    total: int
    # Reads per vehicle class under every filter *except* the class one, so the
    # class filter can show what each choice would give you.
    class_counts: dict[str, int] = Field(default_factory=dict)


# --- Faces ---
class EnrolledFaceOut(BaseModel):
    id: int
    name: str
    has_image: bool
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_serializer("created_at")
    def _ser_created_at(self, v: datetime) -> str | None:
        return _as_utc_iso(v)


class FaceSightingOut(BaseModel):
    id: int
    source_id: int
    name: str | None
    similarity: float
    has_image: bool
    has_frame: bool = False
    timestamp: datetime

    model_config = {"from_attributes": True}

    @field_serializer("timestamp")
    def _ser_timestamp(self, v: datetime) -> str | None:
        return _as_utc_iso(v)
