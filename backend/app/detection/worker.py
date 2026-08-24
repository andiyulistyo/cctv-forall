"""Per-source detection worker.

Runs in its own process (spawned by the manager). For each source it reads
frames, runs YOLO detection + tracking, performs line counting, optional ANPR,
persists events to SQLite, and publishes the latest annotated JPEG + live
counters + status into the shared state for the API to serve.
"""
from __future__ import annotations

import queue
import signal
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from statistics import median

import cv2
import numpy as np
from sqlalchemy import select

from ..config import settings
from ..database import SessionLocal
from ..models import (  # noqa: F401
    DETECTION_CLASSES,
    VEHICLE_CLASSES,
    CountEvent,
    EnrolledFace,
    FaceSighting,
    PlateRead,
    Source,
)
from ..runtime import apply_worker_threads, pin_worker_affinity, threads_per_worker
from .alpr import ALPR
from .capture_gate import CaptureGate
from .detector import Detector, gpu_thread_cap, plan_inference
from .face import FaceRecognizer, face_bbox
from .ffmpeg_log import summarize_ffmpeg_noise
from .frame_store import SharedState
from .line_counter import LineCounter
from .source_reader import (
    CHUNKED_SOURCE_TYPES,
    REALTIME_SOURCE_TYPES,
    BufferedFrameReader,
    SourceReader,
)
from .vehicle_registry import VehicleRegistry

# Per-class BGR colors for drawing.
_CLASS_COLORS = {
    "person": (0, 200, 0),
    "car": (255, 128, 0),
    "motorcycle": (0, 200, 255),
    "truck": (0, 0, 255),
    "bus": (200, 0, 200),
}
_LINE_COLOR = (0, 255, 255)


# How often a worker asks the GPU backend to release cached memory.
_GPU_TRIM_INTERVAL_SEC = 120.0

# How often throughput stats are pushed to the API process.
_STATS_INTERVAL_SEC = 5.0


class _Throughput:
    """Rolling capture / detect / publish rates for one source.

    Published to the shared state so the dashboard (and `GET /sources`) can
    show where a stream is actually spending its time: a low detect_fps means
    inference is the limit, a high dropped_fps means decoding outruns
    processing, and a low capture_fps points at the camera or the network.
    """

    def __init__(self, shared: SharedState, source_id: int):
        self.shared = shared
        self.source_id = source_id
        self.processed = 0
        self.detected = 0
        self._published_mark = 0
        self._decoded_mark = 0
        self._dropped_mark = 0
        self.window_start = time.time()

    def maybe_publish(self, now: float, live_reader=None, preview=None) -> dict | None:
        """Publish a window of rates, or None when the window is not full yet."""
        elapsed = now - self.window_start
        if elapsed < _STATS_INTERVAL_SEC:
            return None
        # Publishing happens on its own thread now, so its rate is sampled from
        # a running total rather than counted here.
        published = preview.published if preview is not None else 0
        stats = {
            "processed_fps": round(self.processed / elapsed, 1),
            "detect_fps": round(self.detected / elapsed, 1),
            "publish_fps": round((published - self._published_mark) / elapsed, 1),
        }
        if live_reader is not None:
            decoded = live_reader.captured
            dropped = live_reader.dropped
            stats["capture_fps"] = round((decoded - self._decoded_mark) / elapsed, 1)
            stats["dropped_fps"] = round((dropped - self._dropped_mark) / elapsed, 1)
            self._decoded_mark, self._dropped_mark = decoded, dropped
        else:
            stats["capture_fps"] = stats["processed_fps"]
        self.shared.set_stats(self.source_id, stats)
        self._published_mark = published
        self.processed = self.detected = 0
        self.window_start = now
        return stats


# A live camera hands frames over at its own pace, so getting fewer than it
# sends means they were lost on the way in: while the worker is busy the socket
# stops being drained, the camera's send buffer fills, and it throws away what
# it cannot push. The decoder only sees the hole afterwards, which is what
# fills the console with "Could not find ref with POC" -- it is asking for a
# reference picture that was dropped upstream.
#
# A feed that keeps up delivers what it advertises, near enough: measured over
# 5s windows on a camera the machine could take, the rate sat at 24.8-25.1 of
# 25 fps. One that does not sat at 20-23.5. So below this share of the stream's
# own rate frames are going missing...
_CAPTURE_LAG_RATIO = 0.95
# ...and above this the feed is whole again. The gap between the two is what
# keeps a rate sitting near the threshold from flapping the status message --
# and with it the source row in the database.
_CAPTURE_OK_RATIO = 0.98
# Stat windows (5s each) the verdict is taken over. The instantaneous rate is
# far too jumpy to judge: a busy scene, a keyframe, or the detection loop
# pausing on a GPU trim all show up as one slow window, and a window that ends
# early even reads *above* the stream rate. The middle of half a minute of them
# is a fair reading of how the feed is doing -- the *median*, not the mean, so
# that a single stalled window cannot convict a stream that is otherwise whole.
_CAPTURE_WINDOWS = 6


class _CaptureHealth:
    """Turns a persistently short capture rate into something actionable.

    What an operator actually sees is a wall of FFmpeg decoder errors (or, with
    ``FFMPEG_LOG_SUMMARY_SECONDS``, a count of them) and a picture that tears.
    Neither says which end is at fault. This does: the stream advertises a
    frame rate, we know how many frames arrive, and a sustained shortfall on a
    real-time source means the frames are being dropped before they get here --
    this machine cannot take the stream at the size the camera is sending it.
    """

    def __init__(self, reader: SourceReader):
        self._reader = reader
        self._window: deque[float] = deque(maxlen=_CAPTURE_WINDOWS)
        self.message: str | None = None

    def update(self, capture_fps: float, frame_shape) -> str | None:
        stream_fps = self._reader.fps
        # An unknown or nonsensical stream rate leaves nothing to compare
        # against; the reader only reports one it believes (see _read_fps).
        if stream_fps <= 0:
            return self.message
        self._window.append(capture_fps)
        if len(self._window) < self._window.maxlen:
            return self.message
        rate = median(self._window)
        if rate < stream_fps * _CAPTURE_LAG_RATIO:
            # Worded once and then left alone: every distinct message is a
            # write to the source row, and re-rendering the same warning with
            # a slightly different number each window would be nothing but
            # churn. The live rate is on the stats panel anyway.
            if self.message is None:
                height, width = frame_shape[:2]
                self.message = (
                    f"receiving {rate:.0f} of {stream_fps:.0f} fps at "
                    f"{width}x{height}: frames are being dropped before they "
                    "reach the decoder because this machine cannot take the "
                    "stream in real time -- use the camera's sub-stream, or "
                    "lower its resolution or frame rate"
                )
        elif rate >= stream_fps * _CAPTURE_OK_RATIO:
            self.message = None
        return self.message


def _ocr_device(detector_device: str) -> str:
    """Device for EasyOCR: explicit setting wins, else follow the detector.

    Only CUDA is inherited. MPS is deliberately not: EasyOCR's recognition
    network is small, parts of it fall back to the CPU anyway, and sharing the
    GPU with YOLO tends to slow both down. The OpenVINO device strings
    ("intel:gpu") mean nothing to EasyOCR at all -- it is a torch model.
    """
    if settings.ocr_device:
        return settings.ocr_device
    if detector_device.startswith("cuda") or detector_device.isdigit():
        return detector_device
    return "cpu"


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


_FACE_KNOWN_COLOR = (255, 200, 0)
_FACE_UNKNOWN_COLOR = (150, 150, 150)
_ZONE_COLOR = (0, 200, 255)
_EVIDENCE_COLOR = (0, 255, 255)


def _zone_px(zone: dict | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Normalized {"a": [x, y], "b": [x, y]} -> a pixel box, or None.

    The corners are normalized so a zone drawn once stays right when the stream
    resolution changes, and sorted here so it does not matter which corner the
    user started the drag from.
    """
    if not zone or "a" not in zone or "b" not in zone:
        return None
    try:
        (ax, ay), (bx, by) = zone["a"], zone["b"]
    except (ValueError, TypeError):
        return None
    x1, x2 = sorted((ax * width, bx * width))
    y1, y2 = sorted((ay * height, by * height))
    return int(x1), int(y1), int(x2), int(y2)


def _zone_overlap(box, zone) -> float:
    """How much of ``box`` lies inside ``zone``, as a fraction of the box area.

    0.0 when they do not meet, 1.0 when the box is wholly inside. Both are
    ``(x1, y1, x2, y2)`` in the same pixel space.
    """
    bx1, by1, bx2, by2 = box
    zx1, zy1, zx2, zy2 = zone
    area = max((bx2 - bx1) * (by2 - by1), 0)
    if area <= 0:
        return 0.0
    iw = max(0, min(bx2, zx2) - max(bx1, zx1))
    ih = max(0, min(by2, zy2) - max(by1, zy1))
    return (iw * ih) / area


def _draw(frame, detections, line_norm, counts: dict, source_cfg: dict, plates: dict, faces=None):
    """Paint the overlay onto ``frame`` in place and return it.

    Takes the counting line and the totals as plain values rather than the
    LineCounter itself: this runs on the preview thread while the detection
    loop is still updating that counter, and ``LineCounter.total()`` walks a
    dict the loop can grow mid-iteration.
    """
    # Plate-reading zone, so the user can see what they configured against the
    # traffic actually going past rather than only on the setup snapshot.
    zone = _zone_px(source_cfg.get("alpr_zone"), frame.shape[1], frame.shape[0])
    if zone is not None:
        cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), _ZONE_COLOR, 1)
        cv2.putText(
            frame, "ANPR", (zone[0] + 4, max(12, zone[1] + 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ZONE_COLOR, 1, cv2.LINE_AA,
        )

    # Counting line
    if line_norm is not None:
        (ax, ay), (bx, by) = line_norm
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
    if line_norm is not None:
        labels = source_cfg.get("direction_labels", {"in": "in", "out": "out"})
        y = 20
        for cls, dirs in counts.items():
            txt = f"{cls}: {labels.get('in','in')}={dirs['in']} {labels.get('out','out')}={dirs['out']}"
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            y += 20

    # Faces
    for (fx, fy, fw, fh, name, sim) in faces or []:
        color = _FACE_KNOWN_COLOR if name else _FACE_UNKNOWN_COLOR
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)
        label = f"{name} {sim:.2f}" if name else "unknown"
        cv2.putText(
            frame, label, (fx, max(12, fy - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
    return frame


class _Preview:
    """Draws and publishes the live view on its own thread.

    Publishing used to sit in the detection loop, one branch after the
    detector, rate limited to MJPEG_FPS. The rate limit made it look
    independent, but a rate limit can only ever slow publishing down: with
    FRAME_STRIDE=1 every pass of that loop runs inference, so the display could
    never be faster than inference, and there were no "frames in between" to
    redraw. Anything that slowed the loop slowed the live view with it -- a
    heavier model, face recognition, or an ANPR budget generous enough to keep
    the OCR thread on the GPU continuously. Past a second or so per frame the
    result is a browser looking at a still image while counting and plate reads
    visibly carry on: a frozen preview on a working system.

    So the two are actually separated here. This thread redraws the newest
    frame at MJPEG_FPS with the most recent boxes, and it takes that frame
    straight from the decoder rather than from the detection loop -- being fed
    by the loop would only move the same coupling one step further out. What it
    does (resize, draw, JPEG encode, socket write) is OpenCV and I/O, all of
    which release the GIL, so it holds its cadence while inference and OCR
    saturate the GPU.

    Paced sources (HLS/YouTube) are the exception. There the detection loop is
    what releases a downloaded burst at the stream's own frame rate, so the
    preview follows the loop; tapping the decoder would play the burst at
    download speed instead.
    """

    def __init__(
        self,
        shared: SharedState,
        source_id: int,
        source_cfg: dict,
        line_norm=None,
        process_width: int = 0,
        fps: int = 15,
        quality: int = 70,
        tap: BufferedFrameReader | None = None,
    ):
        self._shared = shared
        self._source_id = source_id
        self._source_cfg = source_cfg
        self._line_norm = line_norm
        self._process_width = max(0, process_width)
        self._interval = 1.0 / max(1, fps)
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        # The decoder to read the newest frame from, or None when the detection
        # loop feeds us instead (see the class docstring).
        self.tap = tap
        # Overlay state. Each of these is replaced wholesale by the detection
        # loop and a single attribute assignment is atomic, so the worst a race
        # can do is draw one frame with the previous boxes.
        self._frame = None
        self._detections: list = []
        self._faces: list = []
        self._counts: dict = {}
        self._plates: dict = {}
        # Frames published since the worker started; _Throughput samples it.
        self.published = 0
        self._error: type | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="preview", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_frame(self, frame) -> None:
        """Frame to draw on, for sources we do not tap the decoder for."""
        self._frame = frame

    def set_overlay(self, detections, faces, plates: dict) -> None:
        self._detections, self._faces, self._plates = detections, faces, plates

    def set_counts(self, counts: dict) -> None:
        self._counts = counts

    def _prepare(self, frame):
        """The detector's view of a frame, on an array of our own.

        The same downscale detection ran on, so the boxes land where they
        belong -- and a fresh array either way, so drawing cannot reach the
        frame inference or ANPR is still holding.
        """
        if self._process_width and frame.shape[1] > self._process_width:
            new_h = max(1, round(frame.shape[0] * self._process_width / frame.shape[1]))
            return cv2.resize(
                frame, (self._process_width, new_h), interpolation=cv2.INTER_AREA
            )
        return frame.copy()

    def _run(self) -> None:
        seq = 0
        last = None
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                frame = self.tap.latest if self.tap is not None else self._frame
                # Nothing new decoded. The stream endpoint keeps resending what
                # it already has, so re-encoding an identical picture would only
                # burn CPU that detection wants.
                if frame is not None and frame is not last:
                    last = frame
                    annotated = _draw(
                        self._prepare(frame),
                        self._detections,
                        self._line_norm,
                        self._counts,
                        self._source_cfg,
                        self._plates,
                        self._faces,
                    )
                    ok, buf = cv2.imencode(".jpg", annotated, self._encode_params)
                    if ok:
                        seq += 1
                        self.published += 1
                        self._shared.set_frame(self._source_id, buf.tobytes(), seq)
            except Exception as exc:
                # The preview is the one part of a worker that must not be able
                # to take the rest down -- and must not go quiet either: a dead
                # preview thread looks exactly like the frozen stream it would
                # be there to report.
                if type(exc) is not self._error:
                    self._error = type(exc)
                    print(f"[preview] frame dropped: {type(exc).__name__}: {exc}")
            self._stop.wait(max(0.0, self._interval - (time.monotonic() - started)))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def run_worker(source_cfg: dict, shared: SharedState, stop_event, slot: int = 0) -> None:
    # Ctrl-C is delivered to every process attached to the console (a process
    # group signal on POSIX, CTRL_C_EVENT on Windows), so without this a worker
    # dies with a KeyboardInterrupt traceback out of the middle of an inference
    # call. Stopping us is the manager's job -- it sets stop_event and joins --
    # so ignore the signal and leave through the normal path instead.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass

    # Decoding happens here, so this is the process FFmpeg logs from. Install
    # the filter before anything opens a stream.
    summarize_ffmpeg_noise(
        f"source {source_cfg['id']}", settings.ffmpeg_log_summary_seconds
    )

    source_id = source_cfg["id"]
    shared.set_status(source_id, "starting")
    _update_db_status(source_id, "starting")

    # Every source is its own process. Cap the thread pools before touching
    # OpenCV/torch, otherwise each worker grabs all cores and they thrash.
    # An accelerated worker (MPS/CUDA/CoreML/OpenVINO GPU) only pre- and
    # post-processes on the CPU; an OpenVINO *CPU* worker does the whole
    # inference there and needs the full budget.
    model_path = settings.yolo_model_path
    plan = plan_inference(model_path, settings.device, settings.inference_half)
    n_threads = threads_per_worker(
        settings.threads_per_worker,
        settings.expected_streams,
        gpu=not plan.cpu_bound,
        gpu_thread_cap=gpu_thread_cap(plan),
    )
    apply_worker_threads(n_threads)
    pin_worker_affinity(slot, n_threads, settings.worker_cpu_affinity)

    try:
        detector = Detector(
            model_path,
            device=settings.device,
            conf=settings.conf_threshold,
            imgsz=settings.inference_imgsz,
            half=settings.inference_half,
        )
    except Exception as exc:
        shared.set_status(source_id, "error", f"model load failed: {exc}")
        _update_db_status(source_id, "error", f"model load failed: {exc}")
        return

    # Optional ANPR. Constructed here but run on its own thread (_PlateReader)
    # so plate reading never sets the pace of the detection loop.
    alpr: ALPR | None = None
    if settings.alpr_enabled and source_cfg.get("alpr_enabled", True):
        alpr = ALPR(
            languages=settings.ocr_languages,
            device=_ocr_device(detector.device),
            plate_model=settings.plate_model_path,
            plate_imgsz=settings.plate_imgsz,
            plate_conf=settings.plate_conf_threshold,
            min_confidence=settings.plate_min_confidence,
            half=settings.inference_half,
        )
        if not alpr.available:
            alpr = None

    # Optional face recognition
    fstate: _FaceState | None = None
    if settings.face_enabled and source_cfg.get("face_enabled", False):
        rec = FaceRecognizer(
            settings.yunet_model_path,
            settings.sface_model_path,
            det_size=settings.face_det_size,
            max_side=settings.face_max_side,
            backend=settings.face_backend,
        )
        if rec.available:
            fstate = _FaceState(
                rec,
                threshold=settings.face_similarity_threshold,
                cooldown=settings.face_sighting_cooldown_sec,
                log_unknown=settings.face_log_unknown,
            )
            fstate.reload_gallery()

    # Line counter (only if a line is configured)
    counter: LineCounter | None = None
    line = source_cfg.get("line")
    if line and "a" in line and "b" in line:
        counter = LineCounter(
            line_norm=(tuple(line["a"]), tuple(line["b"])),
            hysteresis=settings.count_hysteresis,
        )

    source_type = source_cfg["type"]
    reader = SourceReader(
        source_type,
        source_cfg["url"],
        hwaccel=settings.ffmpeg_hwaccel,
        rtsp_tcp=settings.rtsp_transport_tcp,
        youtube_max_height=settings.youtube_max_height,
    )
    # Decoding always runs in its own thread so the worker never stalls it; how
    # the frames are handed over depends on how the source delivers them.
    live_reader: BufferedFrameReader | None = None
    if source_type in REALTIME_SOURCE_TYPES:
        # Camera sets the pace: keep only the newest frame.
        live_reader = BufferedFrameReader(reader, buffer_frames=1, paced=False)
    elif source_type in CHUNKED_SOURCE_TYPES:
        # HLS/YouTube arrive a segment at a time: buffer the burst and release
        # it at the stream's own frame rate. Backpressure is essential here —
        # an unthrottled decoder thread starves this loop of the GIL.
        #
        # max_latency is what stops that pacing from turning into slow motion
        # when this loop cannot keep up with the stream: past that much drift
        # the reader skips to the newest frame instead of walking through the
        # backlog. Without it a slow loop silently throttles the decoder and
        # the view falls further behind live for as long as the source runs.
        buffer_frames = max(2, int(settings.capture_buffer_seconds * 30))
        live_reader = BufferedFrameReader(
            reader,
            buffer_frames=buffer_frames,
            paced=True,
            backpressure=True,
            max_latency=settings.max_stream_latency_seconds,
        )
    frame_source = live_reader or reader

    enabled_classes = source_cfg.get("enabled_classes", [])

    plate_reader: _PlateReader | None = None
    if alpr is not None:
        plate_reader = _PlateReader(
            alpr,
            source_id,
            max_attempts=settings.alpr_max_attempts,
            attempt_interval=settings.alpr_attempt_interval,
            classes=_alpr_classes(settings.alpr_classes),
            min_vehicle_width=settings.alpr_min_vehicle_width,
            capture_classes=_capture_classes(settings.alpr_capture_classes),
            capture_min_width=settings.alpr_capture_min_width,
            zone=source_cfg.get("alpr_zone"),
            zone_min_overlap=settings.alpr_zone_min_overlap,
            capture_containment=settings.alpr_capture_containment,
            capture_dedupe_seconds=settings.alpr_capture_dedupe_seconds,
            capture_dedupe_overlap=settings.alpr_capture_dedupe_overlap,
            person_sink=_person_sink(source_id, fstate),
            save_frame=settings.alpr_save_frame,
            stationary_seconds=settings.alpr_stationary_seconds,
            reid_gap_seconds=settings.alpr_reid_gap_seconds,
            parked_memory_seconds=settings.alpr_parked_memory_seconds,
        )

    stride = max(1, settings.frame_stride)
    process_width = max(0, settings.process_width)
    last_trim = time.time()
    frame_no = 0
    # Latest detections/faces, handed to the preview thread and redrawn there
    # on every frame so the video stays smooth between detections.
    detections: list = []
    current_faces: list = []

    # The live view runs on its own thread from here on, so how long a pass of
    # this loop takes no longer decides how often the browser gets a frame. A
    # real-time source is tapped straight from the decoder for that; a paced
    # one is released by this loop and has to be fed from it.
    preview = _Preview(
        shared,
        source_id,
        source_cfg,
        line_norm=counter.line_norm if counter is not None else None,
        process_width=process_width,
        fps=settings.mjpeg_fps,
        quality=settings.jpeg_quality,
        tap=live_reader if live_reader is not None and not live_reader.paced else None,
    )
    preview.start()
    stats = _Throughput(shared, source_id)

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

    # Only real-time sources are judged against the stream's frame rate: a
    # chunked one (HLS/YouTube) is deliberately throttled by the jitter buffer,
    # so a low capture rate there means the loop is slow, not the network.
    health = _CaptureHealth(reader) if source_type in REALTIME_SOURCE_TYPES else None
    warning: str | None = None

    try:
        for frame in frame_source.frames(stopped, on_error=on_open_error):
            # First frame after (re)connecting -> mark running. The warning
            # rides along on the status so a degraded feed reads as running
            # with a reason, not as an error.
            report("running", warning)

            frame_no += 1
            stats.processed += 1
            now = time.time()

            # A paced source has no decoder tap: this loop is what releases its
            # frames on the stream's clock, so it hands them to the preview too.
            if preview.tap is None:
                preview.set_frame(frame)

            # Detection is the expensive step and runs every `stride` frames.
            if frame_no % stride != 0:
                continue

            # Detection runs on a downscaled copy when process_width is set;
            # ANPR still crops from the full frame so plate text stays readable.
            # detect_scale converts detection boxes back to full-frame
            # coordinates.
            full_frame = frame
            detect_scale = 1.0
            if process_width and frame.shape[1] > process_width:
                new_h = max(1, round(frame.shape[0] * process_width / frame.shape[1]))
                frame = cv2.resize(
                    frame, (process_width, new_h), interpolation=cv2.INTER_AREA
                )
                detect_scale = full_frame.shape[1] / float(frame.shape[1])

            h, w = frame.shape[:2]

            stats.detected += 1
            detections = detector.track(frame, enabled_classes)

            # Line counting
            if counter is not None:
                counter.set_frame_size(w, h)
                # Ground contact point, not the centroid: see
                # Detection.ground_point.
                crossings = counter.update(
                    [(d.track_id, d.class_name, *d.ground_point) for d in detections]
                )
                if crossings:
                    _persist_crossings(source_id, crossings)
                    totals = counter.total()
                    shared.set_counts(source_id, totals)
                    preview.set_counts(totals)

            # Face recognition. Runs before the ANPR hand-off, not after, so
            # a person captured from this frame can be given the name the
            # recognizer just put to the face inside their box. Nothing else
            # depends on the order of these two.
            current_faces = _run_faces(fstate, frame, source_id) if fstate is not None else []

            # ANPR for vehicles, plus captures for whatever else is in the zone.
            if plate_reader is not None:
                plate_reader.submit(full_frame, detections, detect_scale, current_faces)

            preview.set_overlay(
                detections,
                current_faces,
                plate_reader.plates if plate_reader is not None else {},
            )

            window = stats.maybe_publish(now, live_reader, preview)
            if window is not None and health is not None:
                warning = health.update(window["capture_fps"], full_frame.shape)

            # The MPS caching allocator keeps growing over a long run; trimming
            # occasionally keeps a multi-stream setup within GPU memory.
            if now - last_trim >= _GPU_TRIM_INTERVAL_SEC:
                detector.trim_memory()
                last_trim = now

        shared.set_status(source_id, "stopped")
        _update_db_status(source_id, "stopped")
    except KeyboardInterrupt:
        shared.set_status(source_id, "stopped")
        _update_db_status(source_id, "stopped")
    except Exception as exc:
        shared.set_status(source_id, "error", str(exc))
        _update_db_status(source_id, "error", str(exc))
    finally:
        preview.close()
        if plate_reader is not None:
            plate_reader.close()


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


# Plate jobs waiting for the ANPR thread. Small on purpose: a backlog only
# means reading plates off frames that are already stale, and the vehicle is
# still in view to be retried on a fresher one.
_PLATE_QUEUE_SIZE = 8

# A read this confident is treated as the final answer for that vehicle and
# stops further attempts on it. Re-reading a plate OCR was already sure of
# almost never improves it, and the budget is better spent on the vehicles that
# have no usable read yet. Below this we keep trying and keep the best.
_PLATE_GOOD_ENOUGH_CONF = 0.75

# Filename stand-in for a capture that has no plate text yet. Only the file is
# named this; the row keeps plate_text empty, which is what the UI reads.
_CAPTURE_LABEL = "NOPLATE"


# How much of a vehicle's own size it has to shift before it counts as having
# moved. Detection boxes jitter by a few percent from frame to frame even on a
# stationary car; 15% of the box is well clear of that and still far less than
# any real movement between two detection frames.
_MOTION_FRACTION = 0.15

# Overlap required before a new track id is taken to be a vehicle we already
# know. Two vehicles this close to congruent are one vehicle -- a following car
# in the same lane overlaps far less, and the short re-id window does the rest.
_REID_IOU = 0.6

# How long an ordinary vehicle is remembered after it was last seen. Only
# bounds memory -- whether it can still be recognised is decided by the much
# shorter ALPR_REID_GAP_SECONDS. A vehicle that was parked is kept longer than
# this instead, for ALPR_PARKED_MEMORY_SECONDS.
_FORGET_SEC = 60.0


def _alpr_classes(configured: str) -> tuple[str, ...]:
    """Parse ALPR_CLASSES into the classes we will attempt plate reads on."""
    wanted = [s.strip().lower() for s in configured.split(",") if s.strip()]
    unknown = [n for n in wanted if n not in VEHICLE_CLASSES]
    if unknown:
        print(f"[ALPR] ignoring unknown ALPR_CLASSES entries: {', '.join(unknown)}")
    classes = tuple(n for n in wanted if n in VEHICLE_CLASSES)
    return classes or VEHICLE_CLASSES


def _capture_classes(configured: str) -> tuple[str, ...]:
    """Parse ALPR_CAPTURE_CLASSES into the classes recorded on sight in zone.

    Unlike :func:`_alpr_classes` an empty setting means *none*, not "all": this
    is an opt-in addition, and defaulting it to every class would start writing
    a row for every vehicle that crosses the zone.

    Validated against DETECTION_CLASSES rather than VEHICLE_CLASSES, because
    "person" belongs here too: a capture asks whether something came through the
    zone, and that question is not only about vehicles. A person capture is
    written as a face sighting rather than a plate row -- see
    :func:`_person_sink`.
    """
    wanted = [s.strip().lower() for s in configured.split(",") if s.strip()]
    unknown = [n for n in wanted if n not in DETECTION_CLASSES]
    if unknown:
        print(
            "[ALPR] ignoring unknown ALPR_CAPTURE_CLASSES entries: "
            f"{', '.join(unknown)}"
        )
    return tuple(n for n in wanted if n in DETECTION_CLASSES)


def _face_for_person(box, faces) -> tuple[str | None, float] | None:
    """The face belonging to this person box, or None if none sits inside it.

    Takes the faces the frame has *already* been scanned for rather than running
    the recognizer again: YuNet and SFace have run on this frame by the time a
    capture is queued, so a second pass would buy nothing and cost another
    17-33 ms. It also keeps the recognizer on one thread, which is where the
    ONNX sessions want to stay.

    A named match wins over an unnamed one even if the unnamed face scored
    higher -- an identity is the answer to "who was this", a bare detection is
    not.
    """
    x1, y1, x2, y2 = box
    inside = [
        (name, sim)
        for (fx, fy, fw, fh, name, sim) in faces
        if x1 <= fx + fw / 2.0 <= x2 and y1 <= fy + fh / 2.0 <= y2
    ]
    if not inside:
        return None
    inside.sort(key=lambda t: (t[0] is not None, t[1]), reverse=True)
    return inside[0]


class _PlateReader:
    """Runs ANPR on a background thread instead of in the detection loop.

    One plate read is a plate-detector pass plus an OCR pass, and a busy
    junction keeps several unread vehicles in view at once. Done inline, that
    cost lands on every detection frame, so the *whole* pipeline -- decode,
    tracking, counting, display -- runs at the speed of the OCR. A plate does
    not have to be read on the frame it was seen in, so the crops are handed
    over and the result appears a moment later.

    The queue is short and drops the oldest job when full: ANPR must never push
    back on detection. A vehicle whose crop is dropped is simply tried again on
    a later frame, which is what ``attempt_interval`` already assumes.

    ``zone`` restricts reading to the part of the frame the user marked as the
    one where plates are legible. It is checked against the vehicle's ground
    contact point -- the bottom centre of its box -- because that is where the
    vehicle actually is, while the centre of the box floats higher the taller
    the vehicle and would let a bus qualify from a lane away.

    Per-vehicle state -- attempts, best read, the row to correct -- is held by a
    :class:`VehicleRegistry` rather than keyed on the tracker's id, because a
    vehicle that stops inside the zone is renumbered by the tracker over and
    over and would otherwise be treated as a new arrival every time. See
    ``vehicle_registry`` for why that happens; the consequence here is that a
    stopped or parked car is read while it arrives and then left alone.
    """

    def __init__(
        self,
        alpr: ALPR,
        source_id: int,
        max_attempts: int = 12,
        attempt_interval: int = 3,
        classes: tuple[str, ...] = VEHICLE_CLASSES,
        min_vehicle_width: int = 0,
        capture_classes: tuple[str, ...] = (),
        capture_min_width: int = 0,
        zone: dict | None = None,
        zone_min_overlap: float = 0.5,
        capture_containment: float = 0.99,
        capture_dedupe_seconds: float = 3.0,
        capture_dedupe_overlap: float = 0.8,
        person_sink=None,
        save_frame: bool = False,
        stationary_seconds: float = 20.0,
        reid_gap_seconds: float = 4.0,
        parked_memory_seconds: float = 300.0,
    ):
        self._alpr = alpr
        self._source_id = source_id
        self._max_attempts = max_attempts
        self._attempt_interval = max(1, attempt_interval)
        self._classes = classes
        self._min_vehicle_width = max(0, min_vehicle_width)
        self._zone = zone or None
        # Capture is zone-only: with no zone drawn there is nothing to be
        # "inside", and applying it to the whole frame would record every
        # passing motorcycle instead of the ones at the gate.
        self._capture_classes = tuple(capture_classes)
        if self._capture_classes and self._zone is None:
            print(
                "[ALPR] ALPR_CAPTURE_CLASSES is set but this source has no ANPR "
                "zone; capture stays off until one is drawn."
            )
            self._capture_classes = ()
        self._capture_min_width = max(0, capture_min_width)
        self._zone_min_overlap = min(1.0, max(0.0, zone_min_overlap))
        # Captures are judged on a stricter reading of the same zone than plate
        # reads are. A read wants the earliest frame the plate is legible in and
        # does not care where the rest of the vehicle is; a capture is a record
        # of a passage, and half a vehicle at the edge of the zone is the frame
        # that gets recorded twice. See ``capture_gate``.
        self._gate = CaptureGate(
            min_containment=capture_containment,
            window=capture_dedupe_seconds,
            min_overlap=capture_dedupe_overlap,
        )
        # Where a person capture goes. Vehicles are plate rows; a person is a
        # face sighting, and the worker owns that decision -- this class only
        # decides *whether* something in the zone is worth recording.
        self._person_sink = person_sink
        self._save_frame = save_frame
        # Last exception type reported, so a persistent fault is logged once
        # rather than once per vehicle per frame.
        self._last_error: type | None = None
        self._vehicles = VehicleRegistry(
            reid_gap=reid_gap_seconds,
            reid_iou=_REID_IOU,
            stationary_seconds=stationary_seconds,
            motion_fraction=_MOTION_FRACTION,
            forget_seconds=_FORGET_SEC,
            parked_memory=parked_memory_seconds,
        )
        # track id -> plate text, rebuilt each frame from the registry. Read by
        # the detection loop when it draws; rebinding the attribute is atomic,
        # so the worst a race can do is draw one frame without the new plate.
        self.plates: dict[int, str] = {}
        # Vehicles with a job queued or in progress, so the loop does not pile
        # up several attempts at one vehicle while the first is still running.
        self._pending: set[int] = set()
        self._queue: queue.Queue = queue.Queue(maxsize=_PLATE_QUEUE_SIZE)
        self._thread = threading.Thread(target=self._run, name="alpr", daemon=True)
        self._thread.start()

    def submit(self, frame, detections, scale: float = 1.0, faces=()) -> None:
        """Queue plate crops for the vehicles worth another attempt.

        ``frame`` is the full-resolution frame (plate text has to stay legible)
        and ``scale`` maps detection boxes from the downscaled detection frame
        onto it.

        ``faces`` is what the face recognizer already found on this frame, as
        ``(x, y, w, h, name, similarity)`` in *detection*-frame coordinates. It
        is only used to put a name on a person capture; nothing else here reads
        it, and leaving it empty simply means captures go in unidentified.
        """
        now = time.monotonic()
        # Same space as the boxes below, so the two can be compared at all.
        faces = [
            (fx * scale, fy * scale, fw * scale, fh * scale, name, sim)
            for (fx, fy, fw, fh, name, sim) in faces
        ] if scale != 1.0 else list(faces)
        for det in detections:
            readable = det.class_name in self._classes
            capturable = det.class_name in self._capture_classes
            if not readable and not capturable:
                continue

            box = det.scaled(scale)
            x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
            x2, y2 = int(box.x2), int(box.y2)
            # Registered before any of the skips below: the registry has to see
            # every frame of a vehicle to know whether it is moving, and a
            # vehicle that is skipped for being too small or out of zone is
            # exactly the one that will later be judged on that.
            vehicle = self._vehicles.observe(det.track_id, (x1, y1, x2, y2), now)

            if vehicle.vid in self._pending:
                continue
            if capturable and not readable:
                self._maybe_capture(vehicle, det, frame, (x1, y1, x2, y2), now, faces)
                continue
            if vehicle.best_conf >= _PLATE_GOOD_ENOUGH_CONF:
                continue  # already read well; spend the budget elsewhere
            attempts = vehicle.attempts
            if attempts >= self._max_attempts:
                continue  # give up after several tries

            if x2 - x1 < self._min_vehicle_width:
                # Still too far away to carry plate detail. Deliberately before
                # the attempt is counted: the budget exists for the frames where
                # a read can succeed, and a vehicle approaching the camera earns
                # every one of them by getting closer, not by being in view.
                continue
            if not self._in_zone(frame, (x1, y1, x2, y2)):
                continue  # outside the user's read zone; costs no attempt
            if self._vehicles.is_parked(vehicle, now):
                # Stopped or parked inside the read zone. It was read on the way
                # in, when it was moving; re-reading it now cannot produce a
                # different vehicle, only a different guess at the same plate --
                # which is precisely how one parked car ends up filling the
                # list. Costs no attempt, so it resumes if it pulls away.
                continue

            vehicle.attempts = attempts + 1
            # Attempt at most every few frames per vehicle.
            if attempts % self._attempt_interval != 0:
                continue

            if frame[y1:y2, x1:x2].size == 0:
                continue
            # Anything kept past this call has to be detached from the frame:
            # a numpy view would pin the whole thing for as long as the job sits
            # in the queue, and the evidence frame is drawn on in place.
            if self._save_frame:
                snapshot = frame.copy()
                crop = snapshot[y1:y2, x1:x2]  # a view into our own copy
            else:
                snapshot = None
                crop = frame[y1:y2, x1:x2].copy()
            self._offer(
                vehicle.vid,
                ("read", vehicle.vid, det.track_id, det.class_name, crop, snapshot,
                 (x1, y1, x2, y2), None),
            )

        # Labels follow the vehicle, not the id it happens to carry: a
        # renumbered car keeps the plate already read from it on screen.
        self.plates = self._vehicles.labels()
        self._vehicles.prune(now)

    def _maybe_capture(self, vehicle, det, frame, box, now: float, faces=()) -> None:
        """Record a capture-class vehicle the once it is through the zone.

        Runs a stricter version of the plate read's gauntlet -- zone, size,
        parked -- because it answers a different question: not "can we read this
        plate" but "did this vehicle pass through here". So it fires once per
        *passage* rather than once per attempt, and the row it writes is
        complete on its own with plate_text left empty.

        Three things have to agree that this is a passage we do not already
        have. ``vehicle.captured`` is the cheap one and catches the ordinary
        case. The other two are there because on a zoomed-in camera the tracker
        loses vehicles often enough that identity alone does not hold: the gate
        will not take a vehicle that is only half inside the zone, and it
        remembers what it captured a moment ago so one vehicle reported as two
        nested boxes is not recorded twice. See ``capture_gate`` for the
        measurements behind both.

        Deliberately cheap and non-competitive. It queues no OCR of its own;
        the read, if any, is attempted by the worker thread only when nothing
        else is waiting (see :meth:`_run`), so a busy junction still spends its
        whole OCR budget on the classes in ALPR_CLASSES.
        """
        x1, y1, x2, y2 = box
        if vehicle.captured:
            return  # already recorded; survives the tracker renumbering it
        if x2 - x1 < self._capture_min_width:
            return
        if not self._through_zone(frame, (x1, y1, x2, y2)):
            return
        if self._gate.is_repeat(det.class_name, (x1, y1, x2, y2), now):
            # The same vehicle again under a box we do not recognise as its
            # own -- most often the detector reporting a rider and a machine as
            # one box and then as two. Not marked captured: this vehicle may
            # yet turn out to be a different one that merely arrived where the
            # last one was, and the ledger forgets in a few seconds either way.
            return
        if self._vehicles.is_parked(vehicle, now):
            # Parked in the zone. It was captured when it arrived, or it was
            # already standing there when the worker started -- either way,
            # recording it now would log a vehicle that is not passing through.
            return
        if frame[y1:y2, x1:x2].size == 0:
            return
        if self._save_frame:
            snapshot = frame.copy()
            crop = snapshot[y1:y2, x1:x2]  # a view into our own copy
        else:
            snapshot = None
            crop = frame[y1:y2, x1:x2].copy()
        # Marked only once the job is actually queued. A capture yields to
        # plate reads, so a busy queue drops it -- and marking first would turn
        # "we were busy for one frame" into "this motorcycle was never here".
        # Retrying costs a failed put_nowait per frame, and the moment one gets
        # through the flag stops it.
        if self._offer(
            vehicle.vid,
            ("capture", vehicle.vid, det.track_id, det.class_name, crop, snapshot,
             (x1, y1, x2, y2), _face_for_person((x1, y1, x2, y2), faces)),
            evict=False,
        ):
            vehicle.captured = True
            self._gate.record(det.class_name, (x1, y1, x2, y2), now)

    def _in_zone(self, frame, box) -> bool:
        """Is enough of the vehicle inside the read zone to call it in there?

        Measured as the fraction of the vehicle's own box that falls inside the
        zone, which is the plain reading of "the vehicle is in this area".

        This used to test a single point -- the bottom centre of the box, where
        the vehicle meets the road -- and that turned out to fail in both
        directions on real traffic. Measured on one taxi driving out of the
        sample alley, against a zone the operator had drawn over the near half:

            box 5% inside the zone   -> point inside  -> read, and recorded
            box 19% inside the zone  -> point inside  -> read, and recorded
            box 96% inside the zone  -> point *outside* (the wheels sat just
                                        past the zone's lower edge) -> skipped

        So one car leaving the far end of the zone was recorded three times over
        seven seconds, each time from a different tracker id and sometimes under
        a different class, while the frame where it was actually in the zone was
        the one thrown away. A point cannot express "mostly in the zone", and
        that is the question being asked.

        The reason the old rule preferred the ground point still holds -- the
        centre of a box floats higher the taller the vehicle, so a bus a lane
        away would qualify on its centre. An area test keeps that property
        without the knife edge: that bus overlaps the zone barely at all.
        """
        if self._zone is None:
            return True
        zone = _zone_px(self._zone, frame.shape[1], frame.shape[0])
        if zone is None:
            return True
        return _zone_overlap(box, zone) >= self._zone_min_overlap

    def _through_zone(self, frame, box) -> bool:
        """Is the vehicle all the way inside the read zone, not merely touching it?

        The capture-side counterpart to :meth:`_in_zone`, and stricter for a
        reason worth keeping straight. A plate read wants the *earliest* frame
        the plate is legible in; where the rest of the vehicle happens to be
        does not affect whether the characters can be made out, so a half-in
        vehicle is worth a read. A capture is a claim that a vehicle came past,
        and a vehicle straddling the edge of the zone is the frame that gets
        claimed twice -- once on the way in under one tracker id and again a
        second later under another, by which time it has moved too far for the
        registry to know it is the same machine.

        Unlike ``_in_zone`` this returns False with no zone drawn rather than
        True: capture is zone-only by design, and the constructor has already
        turned the feature off in that case. Guarding here as well keeps the
        rule true of this method on its own.
        """
        if self._zone is None:
            return False
        zone = _zone_px(self._zone, frame.shape[1], frame.shape[0])
        if zone is None:
            return False
        return self._gate.is_through(box, zone)

    def _offer(self, vid: int, job: tuple, evict: bool = True) -> bool:
        """Queue a job; True if it was taken, False if it yielded and was dropped."""
        # Marked pending before the put, not after: the thread can finish the
        # job -- and clear the mark -- before this call returns, and setting it
        # afterwards would leave the vehicle pending for good.
        self._pending.add(vid)
        while True:
            try:
                self._queue.put_nowait(job)
                return True
            except queue.Full:
                if not evict:
                    # A capture waits its turn rather than taking a plate read's
                    # place. This is the whole guarantee that the feature costs
                    # ALPR_CLASSES nothing: when the queue is full it is full of
                    # work that can actually produce a plate.
                    self._pending.discard(vid)
                    return False
                try:
                    stale = self._queue.get_nowait()[1]
                    if stale != vid:
                        self._pending.discard(stale)
                except queue.Empty:
                    pass

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:  # shutdown
                return
            kind, vid, tid, class_name, crop, snapshot, box, match = job
            try:
                if kind == "capture":
                    self._do_capture(vid, tid, class_name, crop, snapshot, box, match)
                    continue
                result = self._alpr.read_plate(crop)
                vehicle = self._vehicles.get(vid)
                if result is not None and vehicle is not None:
                    text, conf, plate_img = result
                    # Only an improvement replaces what we already have: a
                    # confident read from close up must not be overwritten by a
                    # marginal one from the next frame.
                    if conf > vehicle.best_conf:
                        vehicle.best_conf = conf
                        vehicle.text = text
                        row_id = _persist_plate(
                            self._source_id, tid, class_name, text, conf, plate_img,
                            snapshot=snapshot, box=box,
                            row_id=vehicle.row_id,
                        )
                        if row_id is not None:
                            vehicle.row_id = row_id
            except Exception as exc:
                # A failed read is not worth losing the thread over: the vehicle
                # keeps its attempt budget and gets another frame. It is worth
                # saying so, though -- swallowed silently, a broken OCR or a
                # failing write looks exactly like traffic with no readable
                # plates, and there is nothing anywhere to suggest otherwise.
                if type(exc) is not self._last_error:
                    self._last_error = type(exc)
                    print(f"[ALPR] plate read failed: {type(exc).__name__}: {exc}")
            finally:
                self._pending.discard(vid)

    def _do_capture(self, vid, tid, class_name, crop, snapshot, box, match=None) -> None:
        """Write the capture row, then try for a plate only if nothing is waiting.

        The row goes in first and unconditionally: the capture is the point, and
        an OCR failure must not cost us the record of the passage. If the read
        does land it corrects that same row through ``row_id``, which is the
        mechanism the plate path already uses to replace a poor read with a
        better one -- so a motorcycle produces one row either way.
        """
        if class_name == "person":
            # A person has no plate to read and no plate row to live in. Hand it
            # to the worker, which files it as a face sighting under whatever
            # name the recognizer had already put to the face inside this box.
            if self._person_sink is not None:
                name, sim = match if match is not None else (None, 0.0)
                self._person_sink(name, sim, crop, snapshot, box)
            return

        # Detached before the write, because _persist_plate draws the evidence
        # box onto the snapshot and this crop is a view into it -- a 2 px line
        # straight across the edge of what OCR is about to be shown.
        ocr_crop = crop if snapshot is None else crop.copy()
        vehicle = self._vehicles.get(vid)
        row_id = _persist_plate(
            self._source_id, tid, class_name, "", 0.0, crop,
            snapshot=snapshot, box=box,
            row_id=vehicle.row_id if vehicle is not None else None,
        )
        if vehicle is not None and row_id is not None:
            vehicle.row_id = row_id

        # Best effort, and only with the queue idle -- a waiting job is a plate
        # read from ALPR_CLASSES, and this must never be what delays it.
        if vehicle is None or row_id is None or not self._queue.empty():
            return
        result = self._alpr.read_plate(ocr_crop)
        if result is None:
            return
        text, conf, plate_img = result
        if conf <= vehicle.best_conf:
            return
        vehicle.best_conf = conf
        vehicle.text = text
        updated = _persist_plate(
            self._source_id, tid, class_name, text, conf, plate_img,
            snapshot=None, box=box, row_id=row_id,
        )
        if updated is not None:
            vehicle.row_id = updated

    def close(self) -> None:
        # Drop the backlog first, so the sentinel always fits.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)


def _discard_plate_image(image_path: str | None) -> None:
    """Delete an image that has been superseded by a better read."""
    if not image_path:
        return
    try:
        (settings.data_dir / image_path).unlink(missing_ok=True)
    except Exception:
        pass


def _write_image(directory, filename: str, image: np.ndarray) -> str | None:
    """Write an image under ``directory``; return its path relative to data_dir."""
    try:
        fpath = directory / filename
        if not cv2.imwrite(str(fpath), image):
            return None
        return str(fpath.relative_to(settings.data_dir))
    except Exception:
        return None


def _write_evidence_frame(
    filename: str, snapshot: np.ndarray, box: tuple[int, int, int, int] | None, text: str
) -> str | None:
    """Save the full frame with the read vehicle boxed and labelled.

    Drawn in place: the snapshot was copied for this job alone and is dropped
    right after, and the plate crop has already been written by the time this
    runs, so nothing else can see the marks.
    """
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), _EVIDENCE_COLOR, 2)
        text = text or _CAPTURE_LABEL
        cv2.putText(
            snapshot, text, (x1, max(14, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            snapshot, text, (x1, max(14, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _EVIDENCE_COLOR, 1, cv2.LINE_AA,
        )
    return _write_image(settings.frames_dir, filename, snapshot)


def _recent_duplicate(db, source_id: int, text: str) -> PlateRead | None:
    """The row this read is a repeat of, if the text-level window is enabled.

    Off by default. It is the blunt half of duplicate suppression: it catches
    the same plate recorded twice however that happened, but only when OCR read
    it *identically* both times -- and a plate read at low confidence rarely is.
    The vehicle-level suppression in :class:`VehicleRegistry` is what actually
    stops a parked car repeating; this only tidies up behind it.
    """
    window = settings.alpr_duplicate_window_seconds
    if window <= 0:
        return None
    cutoff = _now() - timedelta(seconds=window)
    return db.scalars(
        select(PlateRead)
        .where(
            PlateRead.source_id == source_id,
            PlateRead.plate_text == text,
            PlateRead.timestamp >= cutoff,
        )
        .order_by(PlateRead.timestamp.desc())
        .limit(1)
    ).first()


def _persist_plate(
    source_id, track_id, vehicle_class, text, conf, plate_img: np.ndarray,
    snapshot: np.ndarray | None = None,
    box: tuple[int, int, int, int] | None = None,
    row_id: int | None = None,
) -> int | None:
    """Record the best plate read for one tracked vehicle; return its row id.

    One row per vehicle, corrected in place rather than appended to. A vehicle
    is read several times as it approaches and only the most confident of those
    is the answer; appending would list the same car once per attempt and leave
    the worst read looking exactly as authoritative as the best.

    The row is addressed by ``row_id`` -- the id handed back when it was first
    written -- and never looked up by track id. Trackers restart their numbering
    (the sample feed jumped from 643 back to 1 mid-session), so a track id only
    identifies a vehicle within one run, and matching on it would let a new car
    overwrite an unrelated one recorded earlier.

    The timestamp is not refreshed on a correction: it records when the vehicle
    passed, and reading its plate better does not move that moment.
    """
    stamp = _now()
    # A read is named after the plate, which is what makes it findable on disk.
    # A capture has no plate, and naming every one of them the same thing would
    # have two captures a second apart on one camera resolve to the same file --
    # where the second write silently replaces the first, and superseding it
    # later deletes an image the earlier row still points at. Microseconds are
    # what keeps them distinct.
    label = text or f"{_CAPTURE_LABEL}{stamp:%f}"
    fname = f"{source_id}_{track_id}_{stamp:%Y%m%d_%H%M%S}_{label}.jpg"
    # The crop first: _write_evidence_frame draws on the snapshot, and the crop
    # can be a view into it.
    image_path = _write_image(settings.plates_dir, fname, plate_img)
    frame_path = (
        _write_evidence_frame(fname, snapshot, box, text)
        if snapshot is not None
        else None
    )

    superseded: list[str | None] = []
    db = SessionLocal()
    try:
        row = db.get(PlateRead, row_id) if row_id is not None else None
        if row is None and text:
            # No row of our own yet: with ALPR_DUPLICATE_WINDOW_SECONDS set,
            # one already recorded for this plate on this camera counts as ours.
            # Skipped for a capture: its text is empty, and every unread capture
            # on the camera would match every other one.
            row = _recent_duplicate(db, source_id, text)
            if row is not None and row.confidence >= conf:
                # ...and it is the better read of the two, so it stands as it is
                # and the images just written for this one are the redundant
                # pair. (db.close() still runs; only the discard loop at the end
                # is skipped, and it is done here instead.)
                for path in (image_path, frame_path):
                    _discard_plate_image(path)
                return row.id
        if row is None:
            row = PlateRead(
                source_id=source_id,
                track_id=track_id,
                vehicle_class=vehicle_class,
                plate_text=text,
                confidence=conf,
                image_path=image_path,
                frame_path=frame_path,
                timestamp=_now(),
            )
            db.add(row)
        else:
            row.vehicle_class = vehicle_class
            row.plate_text = text
            row.confidence = conf
            if image_path:
                superseded.append(row.image_path)
                row.image_path = image_path
            if frame_path:
                superseded.append(row.frame_path)
                row.frame_path = frame_path
        db.commit()
        row_id = row.id
    except Exception:
        db.rollback()
        superseded = []  # the row still points at them
    finally:
        db.close()
    for path in superseded:
        _discard_plate_image(path)
    return row_id


# ----------------------- Face recognition -----------------------
_GALLERY_RELOAD_SEC = 30.0


class _FaceState:
    """Holds the recognizer, the enrolled gallery, and per-identity logging
    cooldowns for one running source."""

    def __init__(self, recognizer: FaceRecognizer, threshold: float, cooldown: int, log_unknown: bool):
        self.rec = recognizer
        self.threshold = threshold
        self.cooldown = cooldown
        self.log_unknown = log_unknown
        self.gallery: np.ndarray | None = None
        self.names: list[str] = []
        self.last_reload = 0.0
        self.last_logged: dict[str, float] = {}
        # last_logged is now read from two threads -- the detection loop, via
        # _run_faces, and the ANPR thread, via a person capture. Both ask the
        # same question about the same identity, and they have to get one
        # answer between them or a recognized person walking through the zone
        # is filed twice.
        self._log_lock = threading.Lock()

    def claim_log(self, key: str, now: float) -> bool:
        """Claim the right to log ``key`` now, or False if it is still cooling.

        Claiming and checking are one step on purpose: two threads that both
        looked before either wrote would both see a stale timestamp.
        """
        with self._log_lock:
            if now - self.last_logged.get(key, 0.0) < self.cooldown:
                return False
            self.last_logged[key] = now
            return True

    def reload_gallery(self) -> None:
        db = SessionLocal()
        try:
            rows = db.scalars(select(EnrolledFace)).all()
            embs, names = [], []
            for r in rows:
                if r.embedding:
                    embs.append(np.asarray(r.embedding, dtype=np.float32))
                    names.append(r.name)
            self.gallery = np.stack(embs) if embs else None
            self.names = names
        except Exception:
            pass
        finally:
            db.close()
        self.last_reload = time.time()


def _run_faces(fstate: "_FaceState", frame, source_id: int) -> list:
    # Periodically pick up newly enrolled faces without restarting the worker.
    if time.time() - fstate.last_reload > _GALLERY_RELOAD_SEC:
        fstate.reload_gallery()

    results = []
    for face in fstate.rec.detect(frame):
        emb = fstate.rec.embed(frame, face)
        if emb is None:
            continue
        name, sim = FaceRecognizer.match(emb, fstate.gallery, fstate.names, fstate.threshold)
        x, y, w, h = face_bbox(face)
        results.append((x, y, w, h, name, sim))

        # Log a sighting, throttled per identity (and optionally for unknowns).
        if name is None and not fstate.log_unknown:
            continue
        key = name or "__unknown__"
        if fstate.claim_log(key, time.time()):
            crop = frame[max(0, y): y + h, max(0, x): x + w]
            _persist_sighting(source_id, name, sim, crop)
    return results


def _person_sink(source_id: int, fstate: "_FaceState | None"):
    """Build the callback that files a person captured in the ANPR zone.

    A person capture is a face sighting: it answers "who came through here",
    which is the question the Wajah page already exists to answer. It is filed
    with the name the recognizer put to the face inside the person's box, or
    with no name at all when no face was turned to the camera -- which is the
    case the frame-wide face pass cannot record, because a face it never
    detected is a face it never logs.

    The one thing it must not do is duplicate that frame-wide pass. Both share
    ``_FaceState.claim_log``, so for an identity the face pass would already
    have logged, whichever gets there first wins and the other stands down. An
    *unnamed* capture skips the claim, because with FACE_LOG_UNKNOWN off there
    is nothing on the other side to collide with -- and each person is captured
    once in their own right anyway, via ``Vehicle.captured``.
    """

    def sink(name: str | None, similarity: float, crop, snapshot, box) -> None:
        contested = name is not None or (fstate is not None and fstate.log_unknown)
        if contested and fstate is not None and not fstate.claim_log(
            name or "__unknown__", time.time()
        ):
            return
        _persist_sighting(source_id, name, similarity, crop)

    return sink


def _persist_sighting(source_id: int, name: str | None, similarity: float, crop: np.ndarray) -> None:
    image_path = None
    try:
        ts = _now().strftime("%Y%m%d_%H%M%S_%f")
        label = name or "unknown"
        safe = "".join(c for c in label if c.isalnum() or c in ("-", "_")) or "unknown"
        fpath = settings.faces_dir / f"sight_{source_id}_{safe}_{ts}.jpg"
        if crop.size:
            cv2.imwrite(str(fpath), crop)
            image_path = str(fpath.relative_to(settings.data_dir))
    except Exception:
        image_path = None

    db = SessionLocal()
    try:
        db.add(
            FaceSighting(
                source_id=source_id,
                name=name,
                similarity=float(similarity),
                image_path=image_path,
                timestamp=_now(),
            )
        )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
