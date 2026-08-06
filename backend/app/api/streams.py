"""MJPEG streaming of annotated frames.

Because browsers load these via an <img> tag (which cannot send an
Authorization header), the JWT is accepted as a ``token`` query parameter.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from ..auth import validate_token
from ..config import settings
from ..detection.manager import get_manager

router = APIRouter(prefix="/streams", tags=["streams"])

_BOUNDARY = "frame"

# A tiny gray placeholder JPEG shown before the first real frame arrives.
_PLACEHOLDER = None


def _placeholder_jpeg() -> bytes:
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        import cv2
        import numpy as np

        img = np.full((360, 640, 3), 40, dtype=np.uint8)
        cv2.putText(img, "connecting...", (200, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        ok, buf = cv2.imencode(".jpg", img)
        _PLACEHOLDER = buf.tobytes()
    return _PLACEHOLDER


@router.get("/{source_id}")
def stream(source_id: int, token: str = Query(...)):
    validate_token(token)
    mgr = get_manager()
    interval = 1.0 / max(1, settings.mjpeg_fps)

    def generate():
        while True:
            frame = mgr.get_frame(source_id)
            if frame is None:
                frame = _placeholder_jpeg()
            yield (
                b"--" + _BOUNDARY.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(interval)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
    )
