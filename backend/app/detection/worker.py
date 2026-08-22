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
from datetime import datetime, timezone
from statistics import median

import cv2
import numpy as np
from sqlalchemy import select

from ..config import settings
from ..database import SessionLocal
from ..models import (
    VEHICLE_CLASSES,
    CountEvent,
    EnrolledFace,
    FaceSighting,
    PlateRead,
    Source,
)
from ..runtime import apply_worker_threads, pin_worker_affinity, threads_per_worker
from .alpr import ALPR
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
            zone=source_cfg.get("alpr_zone"),
            save_frame=settings.alpr_save_frame,
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

            # ANPR for vehicles: hand the crops off and move on.
            if plate_reader is not None:
                plate_reader.submit(full_frame, detections, detect_scale)

            # Face recognition
            current_faces = _run_faces(fstate, frame, source_id) if fstate is not None else []

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


def _alpr_classes(configured: str) -> tuple[str, ...]:
    """Parse ALPR_CLASSES into the classes we will attempt plate reads on."""
    wanted = [s.strip().lower() for s in configured.split(",") if s.strip()]
    unknown = [n for n in wanted if n not in VEHICLE_CLASSES]
    if unknown:
        print(f"[ALPR] ignoring unknown ALPR_CLASSES entries: {', '.join(unknown)}")
    classes = tuple(n for n in wanted if n in VEHICLE_CLASSES)
    return classes or VEHICLE_CLASSES


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
    """

    def __init__(
        self,
        alpr: ALPR,
        source_id: int,
        max_attempts: int = 12,
        attempt_interval: int = 3,
        classes: tuple[str, ...] = VEHICLE_CLASSES,
        min_vehicle_width: int = 0,
        zone: dict | None = None,
        save_frame: bool = False,
    ):
        self._alpr = alpr
        self._source_id = source_id
        self._max_attempts = max_attempts
        self._attempt_interval = max(1, attempt_interval)
        self._classes = classes
        self._min_vehicle_width = max(0, min_vehicle_width)
        self._zone = zone or None
        self._save_frame = save_frame
        # Last exception type reported, so a persistent fault is logged once
        # rather than once per vehicle per frame.
        self._last_error: type | None = None
        # track id -> plate text. Written by the ANPR thread, read by the
        # detection loop when it draws; a dict assignment is atomic, so the
        # worst a race can do is draw one frame without the new plate.
        self.plates: dict[int, str] = {}
        # track id -> confidence of the read currently in `plates`. A vehicle is
        # read several times as it approaches, and the later reads are usually
        # the better ones, so the first answer must not be the final one.
        self._best_conf: dict[int, float] = {}
        # track id -> the PlateRead row we wrote for it, so a better read
        # corrects that row instead of adding another.
        self._row_ids: dict[int, int] = {}
        self._attempts: dict[int, int] = {}
        # Tracks with a job queued or in progress, so the loop does not pile up
        # several attempts at one vehicle while the first is still running.
        self._pending: set[int] = set()
        self._queue: queue.Queue = queue.Queue(maxsize=_PLATE_QUEUE_SIZE)
        self._thread = threading.Thread(target=self._run, name="alpr", daemon=True)
        self._thread.start()

    def submit(self, frame, detections, scale: float = 1.0) -> None:
        """Queue plate crops for the vehicles worth another attempt.

        ``frame`` is the full-resolution frame (plate text has to stay legible)
        and ``scale`` maps detection boxes from the downscaled detection frame
        onto it.
        """
        for det in detections:
            if det.class_name not in self._classes:
                continue
            tid = det.track_id
            if tid in self._pending:
                continue
            if self._best_conf.get(tid, 0.0) >= _PLATE_GOOD_ENOUGH_CONF:
                continue  # already read well; spend the budget elsewhere
            attempts = self._attempts.get(tid, 0)
            if attempts >= self._max_attempts:
                continue  # give up after several tries

            box = det.scaled(scale)
            x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
            x2, y2 = int(box.x2), int(box.y2)
            if x2 - x1 < self._min_vehicle_width:
                # Still too far away to carry plate detail. Deliberately before
                # the attempt is counted: the budget exists for the frames where
                # a read can succeed, and a vehicle approaching the camera earns
                # every one of them by getting closer, not by being in view.
                continue
            if not self._in_zone(frame, x1, x2, y2):
                continue  # outside the user's read zone; costs no attempt

            self._attempts[tid] = attempts + 1
            # Attempt at most every few frames per track.
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
            self._offer(tid, (tid, det.class_name, crop, snapshot, (x1, y1, x2, y2)))

    def _in_zone(self, frame, x1: int, x2: int, y2: int) -> bool:
        """Is the vehicle's ground contact point inside the read zone?"""
        if self._zone is None:
            return True
        zone = _zone_px(self._zone, frame.shape[1], frame.shape[0])
        if zone is None:
            return True
        zx1, zy1, zx2, zy2 = zone
        return zx1 <= (x1 + x2) / 2.0 <= zx2 and zy1 <= y2 <= zy2

    def _offer(self, tid: int, job: tuple) -> None:
        # Marked pending before the put, not after: the thread can finish the
        # job -- and clear the mark -- before this call returns, and setting it
        # afterwards would leave the track pending for good.
        self._pending.add(tid)
        while True:
            try:
                self._queue.put_nowait(job)
                return
            except queue.Full:
                try:
                    stale = self._queue.get_nowait()[0]
                    if stale != tid:
                        self._pending.discard(stale)
                except queue.Empty:
                    pass

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:  # shutdown
                return
            tid, class_name, crop, snapshot, box = job
            try:
                result = self._alpr.read_plate(crop)
                if result is not None:
                    text, conf, plate_img = result
                    # Only an improvement replaces what we already have: a
                    # confident read from close up must not be overwritten by a
                    # marginal one from the next frame.
                    if conf > self._best_conf.get(tid, 0.0):
                        self._best_conf[tid] = conf
                        self.plates[tid] = text
                        row_id = _persist_plate(
                            self._source_id, tid, class_name, text, conf, plate_img,
                            snapshot=snapshot, box=box,
                            row_id=self._row_ids.get(tid),
                        )
                        if row_id is not None:
                            self._row_ids[tid] = row_id
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
                self._pending.discard(tid)

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
        cv2.putText(
            snapshot, text, (x1, max(14, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            snapshot, text, (x1, max(14, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _EVIDENCE_COLOR, 1, cv2.LINE_AA,
        )
    return _write_image(settings.frames_dir, filename, snapshot)


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
    fname = f"{source_id}_{track_id}_{_now():%Y%m%d_%H%M%S}_{text}.jpg"
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
        now = time.time()
        if now - fstate.last_logged.get(key, 0.0) >= fstate.cooldown:
            fstate.last_logged[key] = now
            crop = frame[max(0, y): y + h, max(0, x): x + w]
            _persist_sighting(source_id, name, sim, crop)
    return results


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
