"""Face enrollment and recognition-sighting endpoints.

Enrollment loads a lazily-created FaceRecognizer in the API process to compute
the reference embedding. Detection workers keep their own recognizer instance.
Media endpoints accept the JWT via a ``token`` query param (as with plates).
"""
from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import get_current_user, validate_token
from ..config import settings
from ..database import get_db
from ..detection.face import FaceRecognizer, face_bbox
from ..models import EnrolledFace, FaceSighting
from ..schemas import EnrolledFaceOut, FaceSightingOut

router = APIRouter(tags=["faces"])

# Lazily-created recognizer shared by enrollment requests.
_recognizer: FaceRecognizer | None = None


def get_recognizer() -> FaceRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceRecognizer(
            settings.yunet_model_path,
            settings.sface_model_path,
            det_size=settings.face_det_size,
        )
    if not _recognizer.available:
        raise HTTPException(
            503,
            "Face models unavailable. Ensure YuNet/SFace ONNX files exist "
            "(prefetched in Docker) or set YUNET_MODEL / SFACE_MODEL.",
        )
    return _recognizer


# ----------------------- Enrolled faces -----------------------
@router.post("/faces/enroll", response_model=EnrolledFaceOut, dependencies=[Depends(get_current_user)])
async def enroll_face(
    name: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> EnrolledFaceOut:
    rec = get_recognizer()
    raw = await image.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(422, "Could not decode image")

    faces = rec.detect(bgr)
    if not faces:
        raise HTTPException(422, "No face detected in the photo")
    face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
    emb = rec.embed(bgr, face)
    if emb is None:
        raise HTTPException(422, "Could not compute face embedding")

    # Save a crop for the thumbnail.
    image_path = None
    try:
        x, y, w, h = face_bbox(face)
        x, y = max(0, x), max(0, y)
        crop = bgr[y : y + h, x : x + w]
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "face"
        fpath = settings.faces_dir / f"{safe}_{ts}.jpg"
        if crop.size:
            cv2.imwrite(str(fpath), crop)
            image_path = str(fpath.relative_to(settings.data_dir))
    except Exception:
        image_path = None

    ef = EnrolledFace(name=name, embedding=emb.tolist(), image_path=image_path)
    db.add(ef)
    db.commit()
    db.refresh(ef)
    return EnrolledFaceOut(
        id=ef.id, name=ef.name, has_image=bool(ef.image_path), created_at=ef.created_at
    )


@router.get("/faces", response_model=list[EnrolledFaceOut], dependencies=[Depends(get_current_user)])
def list_faces(db: Session = Depends(get_db)) -> list[EnrolledFaceOut]:
    rows = db.scalars(select(EnrolledFace).order_by(EnrolledFace.name, EnrolledFace.id)).all()
    return [
        EnrolledFaceOut(id=r.id, name=r.name, has_image=bool(r.image_path), created_at=r.created_at)
        for r in rows
    ]


@router.delete("/faces/{face_id}", status_code=204, dependencies=[Depends(get_current_user)])
def delete_face(face_id: int, db: Session = Depends(get_db)):
    from fastapi import Response

    ef = db.get(EnrolledFace, face_id)
    if not ef:
        raise HTTPException(404, "Not found")
    if ef.image_path:
        try:
            (settings.data_dir / ef.image_path).unlink(missing_ok=True)
        except Exception:
            pass
    db.delete(ef)
    db.commit()
    return Response(status_code=204)


@router.get("/faces/{face_id}/image")
def face_image(face_id: int, token: str = Query(...), db: Session = Depends(get_db)):
    validate_token(token)
    ef = db.get(EnrolledFace, face_id)
    if not ef or not ef.image_path:
        raise HTTPException(404, "No image")
    fpath = settings.data_dir / ef.image_path
    if not fpath.exists():
        raise HTTPException(404, "Image file missing")
    return FileResponse(str(fpath), media_type="image/jpeg")


# ----------------------- Sightings -----------------------
@router.get("/sightings", response_model=list[FaceSightingOut], dependencies=[Depends(get_current_user)])
def list_sightings(
    source: int | None = Query(None),
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = Query(None),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db),
) -> list[FaceSightingOut]:
    stmt = select(FaceSighting).order_by(FaceSighting.timestamp.desc()).limit(limit)
    if source is not None:
        stmt = stmt.where(FaceSighting.source_id == source)
    if from_ is not None:
        stmt = stmt.where(FaceSighting.timestamp >= from_)
    if to is not None:
        stmt = stmt.where(FaceSighting.timestamp <= to)
    rows = db.scalars(stmt).all()
    return [
        FaceSightingOut(
            id=s.id,
            source_id=s.source_id,
            name=s.name,
            similarity=s.similarity,
            has_image=bool(s.image_path),
            timestamp=s.timestamp,
        )
        for s in rows
    ]


@router.get("/sightings/{sighting_id}/image")
def sighting_image(sighting_id: int, token: str = Query(...), db: Session = Depends(get_db)):
    validate_token(token)
    s = db.get(FaceSighting, sighting_id)
    if not s or not s.image_path:
        raise HTTPException(404, "No image")
    fpath = settings.data_dir / s.image_path
    if not fpath.exists():
        raise HTTPException(404, "Image file missing")
    return FileResponse(str(fpath), media_type="image/jpeg")
