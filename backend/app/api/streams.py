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
from ..detection.manager import SHUTTING_DOWN, get_manager

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


# Re-send the current frame at least this often, so a paused source still
# looks like a live connection to the browser and to any proxy in between.
_KEEPALIVE_SEC = 2.0


@router.get("/{source_id}")
def stream(source_id: int, token: str = Query(...)):
    validate_token(token)
    mgr = get_manager()
    interval = 1.0 / max(1, settings.mjpeg_fps)
    # Poll finer than the frame interval so a new frame goes out as soon as it
    # exists instead of waiting for the next tick — sampling on a fixed timer
    # that is not aligned with the worker is what makes an otherwise steady
    # stream look jerky.
    poll = max(0.005, interval / 4)

    def generate():
        last_seq = -1
        last_sent = 0.0
        while not SHUTTING_DOWN.is_set():
            # Cheap integer read; the JPEG itself (tens of KB, transferred over
            # the manager socket) is only pulled when it has actually changed.
            seq = mgr.shared.get_frame_seq(source_id)
            now = time.time()
            if seq == last_seq and now - last_sent < _KEEPALIVE_SEC:
                time.sleep(poll)
                continue

            frame = mgr.get_frame(source_id) if seq else None
            if frame is None:
                frame = _placeholder_jpeg()
            last_seq, last_sent = seq, now
            yield (
                b"--" + _BOUNDARY.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            # Never exceed the configured frame rate, however fast the worker
            # publishes.
            time.sleep(interval)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
    )
