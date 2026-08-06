"""Per-source detection worker.

Runs in its own process (spawned by the manager). For each source it reads
frames, runs YOLO detection + tracking, performs line counting, optional ANPR,
persists events to SQLite, and publishes the latest annotated JPEG + live
counters + status into the shared state for the API to serve.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import cv2
import numpy as np

from ..config import settings
from ..database import SessionLocal
from ..models import VEHICLE_CLASSES, CountEvent, PlateRead, Source
from .alpr import ALPR
from .detector import Detector
from .frame_store import SharedState
from .line_counter import LineCounter
from .source_reader import SourceReader

# Per-class BGR colors for drawing.
_CLASS_COLORS = {
    "person": (0, 200, 0),
    "car": (255, 128, 0),
    "motorcycle": (0, 200, 255),
    "truck": (0, 0, 255),
    "bus": (200, 0, 200),
}
_LINE_COLOR = (0, 255, 255)


def _now() -> datetime:
    # Naive UTC to match models._utcnow / retention comparisons.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _update_db_status(source_id: int, status: str, message: str | None = None) -> None:
    db = SessionLocal()
    try:
        src = db.get(Source, source_id)
        if src is not None:
            src.status = status
            src.status_message = message
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _draw(frame, detections, counter: LineCounter | None, source_cfg: dict, plates: dict):
    # Counting line
    if counter is not None:
        (ax, ay), (bx, by) = counter.line_norm
        h, w = frame.shape[:2]
        cv2.line(frame, (int(ax * w), int(ay * h)), (int(bx * w), int(by * h)), _LINE_COLOR, 2)

    for det in detections:
        color = _CLASS_COLORS.get(det.class_name, (255, 255, 255))
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        label = f"{det.class_name} #{det.track_id}"
        plate = plates.get(det.track_id)
        if plate:
            label += f" [{plate}]"
        cv2.putText(
            frame, label, (p1[0], max(12, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    # Counts overlay
    if counter is not None:
        labels = source_cfg.get("direction_labels", {"in": "in", "out": "out"})
        y = 20
        for cls, dirs in counter.total().items():
            txt = f"{cls}: {labels.get('in','in')}={dirs['in']} {labels.get('out','out')}={dirs['out']}"
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            y += 20
    return frame


def run_worker(source_cfg: dict, shared: SharedState, stop_event) -> None:
    source_id = source_cfg["id"]
    shared.set_status(source_id, "starting")
    _update_db_status(source_id, "starting")

    try:
        detector = Detector(
            settings.yolo_model,
            device=settings.device,
            conf=settings.conf_threshold,
            imgsz=settings.inference_imgsz,
        )
    except Exception as exc:
        shared.set_status(source_id, "error", f"model load failed: {exc}")
        _update_db_status(source_id, "error", f"model load failed: {exc}")
        return

    # Optional ANPR
    alpr: ALPR | None = None
    if settings.alpr_enabled and source_cfg.get("alpr_enabled", True):
        alpr = ALPR(
            languages=settings.ocr_languages,
            use_gpu=(detector.device != "cpu"),
            plate_model=settings.plate_model,
        )
        if not alpr.available:
            alpr = None

    # Line counter (only if a line is configured)
    counter: LineCounter | None = None
    line = source_cfg.get("line")
    if line and "a" in line and "b" in line:
        counter = LineCounter(line_norm=(tuple(line["a"]), tuple(line["b"])))

    reader = SourceReader(source_cfg["type"], source_cfg["url"])
    enabled_classes = source_cfg.get("enabled_classes", [])

    stride = max(1, settings.frame_stride)
    frame_no = 0
    plates_by_track: dict[int, str] = {}
    alpr_attempts: dict[int, int] = {}
    last_publish = 0.0
    publish_interval = 1.0 / max(1, settings.mjpeg_fps)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]

    def stopped() -> bool:
        return stop_event.is_set()

    # Track last reported status so we only write on change (avoids DB churn
    # while a source keeps failing to connect).
    current_status = {"v": None}

    def report(state: str, message: str | None = None) -> None:
        if current_status["v"] != (state, message):
            current_status["v"] = (state, message)
            shared.set_status(source_id, state, message)
            _update_db_status(source_id, state, message)

    def on_open_error(message: str) -> None:
        report("error", message)

    try:
        for frame in reader.frames(stopped, on_error=on_open_error):
            # First frame after (re)connecting -> mark running.
            report("running")

            frame_no += 1
            if frame_no % stride != 0:
                continue

            h, w = frame.shape[:2]
            detections = detector.track(frame, enabled_classes)

            # Line counting
            if counter is not None:
                counter.set_frame_size(w, h)
                crossings = counter.update(
                    [(d.track_id, d.class_name, *d.centroid) for d in detections]
                )
                if crossings:
                    _persist_crossings(source_id, crossings)
                    shared.set_counts(source_id, counter.total())

            # ANPR for vehicles
            if alpr is not None:
                _run_alpr(alpr, frame, detections, plates_by_track, alpr_attempts, source_id)

            # Annotate + publish (rate limited)
            now = time.time()
            if now - last_publish >= publish_interval:
                annotated = _draw(frame.copy(), detections, counter, source_cfg, plates_by_track)
                ok, buf = cv2.imencode(".jpg", annotated, encode_params)
                if ok:
                    shared.set_frame(source_id, buf.tobytes())
                last_publish = now

        shared.set_status(source_id, "stopped")
        _update_db_status(source_id, "stopped")
    except Exception as exc:
        shared.set_status(source_id, "error", str(exc))
        _update_db_status(source_id, "error", str(exc))


def _persist_crossings(source_id: int, crossings) -> None:
    db = SessionLocal()
    try:
        for c in crossings:
            db.add(
                CountEvent(
                    source_id=source_id,
                    class_name=c.class_name,
                    direction=c.direction,
                    track_id=c.track_id,
                    timestamp=_now(),
                )
            )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _run_alpr(alpr: ALPR, frame, detections, plates_by_track, alpr_attempts, source_id: int) -> None:
    for det in detections:
        if det.class_name not in VEHICLE_CLASSES:
            continue
        tid = det.track_id
        if tid in plates_by_track:
            continue  # already read a plate for this vehicle
        attempts = alpr_attempts.get(tid, 0)
        if attempts >= 12:
            continue  # give up after several tries to save CPU
        # Attempt at most every few frames per track.
        alpr_attempts[tid] = attempts + 1
        if attempts % 3 != 0:
            continue

        x1, y1 = max(0, int(det.x1)), max(0, int(det.y1))
        x2, y2 = int(det.x2), int(det.y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        result = alpr.read_plate(crop)
        if result is None:
            continue
        text, conf, plate_img = result
        plates_by_track[tid] = text
        _persist_plate(source_id, tid, det.class_name, text, conf, plate_img)


def _persist_plate(source_id, track_id, vehicle_class, text, conf, plate_img: np.ndarray) -> None:
    image_path = None
    try:
        ts = _now().strftime("%Y%m%d_%H%M%S")
        fname = f"{source_id}_{track_id}_{ts}_{text}.jpg"
        fpath = settings.plates_dir / fname
        cv2.imwrite(str(fpath), plate_img)
        image_path = str(fpath.relative_to(settings.data_dir))
    except Exception:
        image_path = None

    db = SessionLocal()
    try:
        db.add(
            PlateRead(
                source_id=source_id,
                track_id=track_id,
                vehicle_class=vehicle_class,
                plate_text=text,
                confidence=conf,
                image_path=image_path,
                timestamp=_now(),
            )
        )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
