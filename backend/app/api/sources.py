"""Source CRUD + lifecycle (start/stop), snapshot and line configuration."""
from __future__ import annotations

import cv2
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..detection.manager import get_manager
from ..detection.source_reader import resolve_stream_url
from ..models import Source
from ..schemas import (
    LineUpdate,
    SourceCreate,
    SourceListResponse,
    SourceOut,
    SourceUpdate,
)

router = APIRouter(prefix="/sources", tags=["sources"], dependencies=[Depends(get_current_user)])


def build_source_cfg(src: Source) -> dict:
    """Plain, picklable config passed to a worker process."""
    return {
        "id": src.id,
        "name": src.name,
        "type": src.type,
        "url": src.url,
        "enabled_classes": list(src.enabled_classes or []),
        "line": src.line,
        "direction_labels": src.direction_labels or {"in": "in", "out": "out"},
        "alpr_enabled": bool(src.alpr_enabled),
    }


def _to_out(src: Source, mgr) -> SourceOut:
    out = SourceOut.model_validate(src)
    # Reflect live status from the manager if the worker is running.
    if mgr.is_running(src.id):
        st = mgr.shared.get_status(src.id)
        if st:
            out.status, out.status_message = st[0], st[1]
    return out


@router.get("", response_model=SourceListResponse)
def list_sources(db: Session = Depends(get_db)) -> SourceListResponse:
    mgr = get_manager()
    sources = db.scalars(select(Source).order_by(Source.id)).all()
    items = [_to_out(s, mgr) for s in sources]
    active = len(mgr.running_ids())
    return SourceListResponse(sources=items, total=len(items), active=active)


@router.post("", response_model=SourceOut, status_code=201)
def create_source(payload: SourceCreate, db: Session = Depends(get_db)) -> SourceOut:
    try:
        payload.validate_semantics()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    src = Source(
        name=payload.name,
        type=payload.type,
        url=payload.url,
        enabled_classes=payload.enabled_classes,
        alpr_enabled=1 if payload.alpr_enabled else 0,
        status="stopped",
    )
    db.add(src)
    db.commit()
    db.refresh(src)
    return _to_out(src, get_manager())


@router.get("/{source_id}", response_model=SourceOut)
def get_source(source_id: int, db: Session = Depends(get_db)) -> SourceOut:
    src = db.get(Source, source_id)
    if not src:
        raise HTTPException(404, "Source not found")
    return _to_out(src, get_manager())


@router.put("/{source_id}", response_model=SourceOut)
def update_source(source_id: int, payload: SourceUpdate, db: Session = Depends(get_db)) -> SourceOut:
    src = db.get(Source, source_id)
    if not src:
        raise HTTPException(404, "Source not found")
    if payload.name is not None:
        src.name = payload.name
    if payload.url is not None:
        src.url = payload.url
    if payload.enabled_classes is not None:
        from ..models import DETECTION_CLASSES

        bad = [c for c in payload.enabled_classes if c not in DETECTION_CLASSES]
        if bad:
            raise HTTPException(422, f"unknown classes: {bad}")
        src.enabled_classes = payload.enabled_classes
    if payload.alpr_enabled is not None:
        src.alpr_enabled = 1 if payload.alpr_enabled else 0
    db.commit()
    db.refresh(src)
    return _to_out(src, get_manager())


@router.delete("/{source_id}", status_code=204)
def delete_source(source_id: int, db: Session = Depends(get_db)) -> Response:
    src = db.get(Source, source_id)
    if not src:
        raise HTTPException(404, "Source not found")
    mgr = get_manager()
    if mgr.is_running(source_id):
        mgr.stop(source_id)
    db.delete(src)
    db.commit()
    return Response(status_code=204)


@router.put("/{source_id}/line", response_model=SourceOut)
def set_line(source_id: int, payload: LineUpdate, db: Session = Depends(get_db)) -> SourceOut:
    src = db.get(Source, source_id)
    if not src:
        raise HTTPException(404, "Source not found")
    src.line = {"a": payload.line.a, "b": payload.line.b}
    if payload.direction_labels is not None:
        src.direction_labels = {
            "in": payload.direction_labels.in_,
            "out": payload.direction_labels.out,
        }
    db.commit()
    db.refresh(src)
    # Restart the worker so the new line takes effect immediately.
    mgr = get_manager()
    if mgr.is_running(source_id):
        mgr.stop(source_id)
        mgr.start(build_source_cfg(src))
    return _to_out(src, mgr)


@router.post("/{source_id}/start", response_model=SourceOut)
def start_source(source_id: int, db: Session = Depends(get_db)) -> SourceOut:
    src = db.get(Source, source_id)
    if not src:
        raise HTTPException(404, "Source not found")
    mgr = get_manager()
    mgr.start(build_source_cfg(src))
    src.status = "starting"
    db.commit()
    db.refresh(src)
    return _to_out(src, mgr)


@router.post("/{source_id}/stop", response_model=SourceOut)
def stop_source(source_id: int, db: Session = Depends(get_db)) -> SourceOut:
    src = db.get(Source, source_id)
    if not src:
        raise HTTPException(404, "Source not found")
    mgr = get_manager()
    mgr.stop(source_id)
    src.status = "stopped"
    src.status_message = None
    db.commit()
    db.refresh(src)
    return _to_out(src, mgr)


@router.get("/{source_id}/snapshot")
def snapshot(source_id: int, db: Session = Depends(get_db)) -> Response:
    """Return a single JPEG frame for line drawing.

    If the worker is running, reuse its latest annotated frame; otherwise grab
    one raw frame directly from the source.
    """
    src = db.get(Source, source_id)
    if not src:
        raise HTTPException(404, "Source not found")

    mgr = get_manager()
    if mgr.is_running(source_id):
        frame = mgr.get_frame(source_id)
        if frame:
            return Response(content=frame, media_type="image/jpeg")

    jpeg = _grab_single_frame(src.type, src.url)
    if jpeg is None:
        raise HTTPException(503, "Could not read a frame from the source")
    return Response(content=jpeg, media_type="image/jpeg")


def _grab_single_frame(source_type: str, url: str) -> bytes | None:
    try:
        stream_url = resolve_stream_url(source_type, url)
    except Exception:
        return None
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return None
        last = None
        for _ in range(5):  # skip a few frames to get a stable image
            ok, frame = cap.read()
            if ok and frame is not None:
                last = frame
        if last is None:
            return None
        ok, buf = cv2.imencode(".jpg", last)
        return buf.tobytes() if ok else None
    finally:
        cap.release()
