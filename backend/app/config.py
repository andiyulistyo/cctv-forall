"""Application configuration, loaded from environment variables / .env."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root inside the container/host: .../backend
BASE_DIR = Path(__file__).resolve().parent.parent
# Default data directory (mounted as a volume in Docker) holds the SQLite DB,
# downloaded model weights and cropped plate images.
DEFAULT_DATA_DIR = BASE_DIR.parent / "data"


class Settings(BaseSettings):
    # Look for .env both at the project root and in the current directory, so
    # the same file works whether the app is started from the repo root or from
    # backend/ (later entries win). Real environment variables always win.
    model_config = SettingsConfigDict(
        env_file=(str(BASE_DIR.parent / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General ---
    app_name: str = "Detection Dashboard"
    data_dir: Path = DEFAULT_DATA_DIR

    # --- Auth (single admin) ---
    admin_username: str = "admin"
    admin_password: str = "admin"  # override in production via env
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 1 day

    # --- Detection ---
    # Ultralytics weights. "yolo11n.pt" auto-downloads on first use. It may
    # also be an accelerator-specific export:
    #   - CoreML bundle  ("yolo11n.mlpackage",        scripts/export_coreml.py)
    #   - OpenVINO IR    ("yolo11n_openvino_model/",  scripts/export_openvino.py)
    # Both have a FIXED input size, so inference_imgsz must match the export.
    #
    # "yolo12*.pt" is accepted too and is what the CUDA profile uses (see
    # .env.nvidia.example). It stays out of this default on purpose: v12's
    # area-attention blocks are markedly more expensive on a CPU, and this
    # default is the one CPU-only and Docker installs fall back to.
    yolo_model: str = "yolo11n.pt"
    # Force a device ("cpu" / "cuda" / "mps" / "0", or "intel:gpu" / "intel:cpu"
    # for an OpenVINO model). Empty string => auto-detect: cuda -> mps -> cpu
    # for torch weights, Intel iGPU (else the OpenVINO CPU plugin) for an
    # OpenVINO model.
    device: str = ""
    # Inference image size (smaller = faster on CPU).
    inference_imgsz: int = 640
    # Detection confidence threshold.
    conf_threshold: float = 0.35
    # Process 1 out of every N frames (frame skipping). 1 = every frame.
    frame_stride: int = 2
    # Run inference in fp16. Free speed-up on MPS/CUDA, ignored on CPU.
    inference_half: bool = False
    # How far past the counting line, as a fraction of frame height, a vehicle
    # must travel before that side counts as reached.
    #
    # Detection boxes jitter, so without a band an object near the line gets
    # nudged back and forth across it and is counted every time. Measured on the
    # sample junction feed, that produced 8.5% more crossings than there were
    # cars and 18.5% more than there were trucks. Inside the band a vehicle's
    # side is simply undecided, so jitter registers nothing.
    #
    # Keep it small: it is also the distance a vehicle must still travel after
    # crossing, so a line drawn very close to the frame edge with a large band
    # can miss vehicles that leave the picture first. 0 disables it.
    count_hysteresis: float = 0.02
    # Downscale frames to this width before detection/annotation (0 = keep the
    # source resolution). 1280 or 960 cuts a lot of work on 1080p/4K CCTV;
    # ANPR still crops from the full-resolution frame.
    process_width: int = 0

    # --- Performance / threading ---
    # CPU threads per worker process (0 = auto from performance-core count).
    threads_per_worker: int = 0
    # How many sources you plan to run at once; used to split CPU cores between
    # worker processes when threads_per_worker is auto.
    expected_streams: int = 4
    # Give each worker its own slice of the CPU ("auto" / "off"). Thread-count
    # env vars do not reach every backend -- the OpenVINO CPU plugin schedules
    # on TBB and only honours the process affinity mask. No-op on macOS.
    worker_cpu_affinity: str = "auto"
    # Hardware video decoding for network streams:
    #   "auto"         let OpenCV pick (D3D11VA on Windows, VAAPI on Linux,
    #                  VideoToolbox on macOS) and fall back to software itself
    #   "off" / ""     software decoding
    #   "d3d11va" / "qsv" / "vaapi"   force one backend
    #   "videotoolbox" / "cuda"       passed to FFmpeg as a capture option
    # Falls back to software automatically if the stream cannot be opened.
    ffmpeg_hwaccel: str = "auto"
    # Force RTSP over TCP (far fewer corrupt frames than the UDP default).
    rtsp_transport_tcp: bool = True
    # FFmpeg writes one line straight to stderr for every picture it cannot
    # decode ("Could not find ref with POC 44"), which buries every other log
    # line as soon as a feed starts losing frames. A worker collapses those
    # into a single summary line this often. 0 prints them raw.
    ffmpeg_log_summary_seconds: float = 30.0
    # Cap the YouTube stream resolution. A live YouTube feed is often 1080p or
    # 4K; decoding that costs far more than the detection itself and buys no
    # accuracy at imgsz=640.
    youtube_max_height: int = 720
    # Jitter buffer for chunked sources (HLS/YouTube). They arrive in bursts of
    # a whole segment at a time; buffering this many seconds and releasing at
    # the stream's own frame rate turns the bursts back into smooth video.
    capture_buffer_seconds: float = 2.0
    # How far behind real time a live stream may drift before the reader stops
    # replaying the backlog and skips to the newest decoded frame.
    #
    # The jitter buffer above paces frames out at the stream's own rate, which
    # is only correct while detection keeps up. When it does not, pacing turns
    # a live feed into slow motion that falls further behind for as long as the
    # app runs -- and the drop counters stay at zero throughout, because the
    # decoder is being blocked rather than outrun. This is the ceiling on that
    # drift. Keep it small: catching up means jumping over the frames in
    # between, so a bigger budget buys one large lurch, not smoothness. Set to
    # 0 to disable skipping (only sensible for a recorded source, where
    # processing every frame matters more than staying current).
    max_stream_latency_seconds: float = 0.5

    # --- ANPR ---
    alpr_enabled: bool = True
    # Optional dedicated plate-detector weights. If empty, a heuristic ROI
    # (lower part of the vehicle box) is used before OCR. The heuristic is the
    # biggest source of misreads, so on a machine with a GPU to spare a real
    # detector is worth far more than any OCR tuning.
    plate_model: str = ""
    # Inference size for the plate detector. It runs on a vehicle crop, not a
    # full frame, so 320 is plenty -- 640 just upscales a small box.
    plate_imgsz: int = 320
    # Confidence threshold for the plate detector. Lower than the vehicle one:
    # a missed plate costs a whole read, while a false box is thrown out by the
    # plate-format check after OCR anyway.
    plate_conf_threshold: float = 0.25
    # Minimum EasyOCR confidence before a plate is stored. Without a floor,
    # anything matching the Indonesian plate pattern is persisted no matter how
    # unsure the OCR was, and a distant/blurry plate reliably produces a
    # confident-looking string at ~0.08 that is simply wrong. A wrong plate is
    # worse than no plate, so reads below this are dropped and the vehicle is
    # retried on a later frame.
    plate_min_confidence: float = 0.20
    # How many frames to keep trying a plate for one tracked vehicle, and how
    # often within those attempts to actually run OCR. The defaults are the
    # CPU-era budget; on a GPU both can be far more generous, which is what
    # decides whether a vehicle is caught on the one frame where its plate is
    # legible (see .env.nvidia.example).
    alpr_max_attempts: int = 12
    alpr_attempt_interval: int = 3
    # Which tracked classes are worth reading a plate from.
    #
    # Motorcycles are left out by default, and not for lack of interest: an
    # Indonesian motorcycle plate is roughly half the width of a car plate and
    # sits low and often angled, so on a typical overview camera it lands on a
    # few dozen pixels and OCR returns nothing. Measured on the sample junction
    # feed, motorcycles were 77% of the vehicles crossing the line and produced
    # zero reads -- while still taking 77% of the OCR budget away from the cars
    # and trucks that can be read. Add "motorcycle" back if your camera is
    # close enough to make their plates legible.
    alpr_classes: str = "car,truck,bus"
    # Classes recorded on sight *inside the ANPR zone*, whether or not a plate
    # can be read from them. Empty (the default) keeps the old behaviour, where
    # nothing is written unless OCR produced text.
    #
    # This exists for motorcycles. They are excluded from alpr_classes above
    # because their plates are usually unreadable on an overview camera -- but
    # "we could not read its plate" is not the same as "it was not there", and
    # on a gate or alley camera the passage itself is the thing worth keeping.
    # A capture writes the vehicle crop and the full frame and leaves
    # plate_text empty; if a plate is read afterwards the same row is filled in
    # rather than a second one appended.
    #
    # Strictly zone-only, by request and by design: with no alpr_zone drawn on
    # a source, capture is off for that source rather than applied to the whole
    # frame. Nothing here can take OCR budget away from alpr_classes -- a
    # capture never evicts a queued plate read, and the opportunistic read that
    # follows one runs only when no read is waiting.
    #
    # A class listed in alpr_classes as well is read, not captured: it already
    # has the full attempt budget, and capturing it too would write a row for
    # every vehicle that entered the zone.
    alpr_capture_classes: str = ""
    # Minimum width, in full-resolution pixels, of a box before it is captured.
    #
    # Deliberately separate from alpr_min_vehicle_width: that floor is about
    # whether a *plate* can be resolved, and at 160 px it rejects almost every
    # motorcycle on an overview camera -- which would leave this feature
    # capturing nothing. A capture only has to show the vehicle, so the floor
    # is much lower. 0 disables it.
    alpr_capture_min_width: int = 48
    # Minimum width, in full-resolution pixels, of a vehicle box before a plate
    # read is attempted on it.
    #
    # Without this the attempt budget is spent blind: a vehicle is tracked from
    # the moment it appears at the far end of the frame -- smallest, least
    # legible -- so the budget is typically exhausted before it comes close
    # enough to read. A plate is roughly a quarter of the width of the vehicle
    # around it, so 160 px of vehicle is about 40 px of plate: already marginal,
    # and a useful floor rather than a target. Raise it until only vehicles in
    # the readable part of your frame qualify; lower it for a close-up camera.
    # Frames below the threshold cost no attempt, so the budget survives for
    # where it can work.
    alpr_min_vehicle_width: int = 160
    # --- Duplicate suppression ---
    #
    # A vehicle that stops inside the read zone -- waiting at a gate, or parked
    # in shot all evening -- is the one case where the attempt budget does not
    # end the reading. The tracker keeps losing a motionless vehicle and giving
    # it a new id, and a new id used to mean a new budget and a new row, so one
    # parked car filled the plate list with a fresh (differently misread) entry
    # every few minutes.
    #
    # How long a vehicle must hold still before it is treated as stopped and
    # left alone. It is still read on the way in, while it is moving; only the
    # endless re-reading afterwards is dropped. It resumes the moment the
    # vehicle pulls away. 0 disables the check -- reasonable only where nothing
    # ever stops in the zone.
    alpr_stationary_seconds: float = 20.0
    # How long a vehicle may be missing before a box appearing in the same place
    # counts as a *different* vehicle rather than the same one renumbered. This
    # is the whole safety margin: a tracker blink lasts a frame or two, whereas
    # one car leaving and the next pulling into the same spot takes far longer.
    # Raise it if a parked car still slips through as new; lower it if two cars
    # queueing in the same spot get merged into one row. 0 disables re-id, which
    # brings back the duplicates.
    alpr_reid_gap_seconds: float = 4.0
    # How long a vehicle already established as parked stays recognisable after
    # the detector stops seeing it at all. A motionless car does not just get
    # renumbered, it disappears: a static shape against a static background
    # drops below the confidence threshold and comes back minutes later, which
    # is why the reads in the reported case were 2-4 minutes apart. Without this
    # every reappearance is a new arrival. It only applies to vehicles that had
    # already held still for ALPR_STATIONARY_SECONDS, so a car pausing at a gate
    # is unaffected. Lower it where parking spots turn over quickly.
    alpr_parked_memory_seconds: float = 300.0
    # Optional second net, on the text rather than the vehicle: within this many
    # seconds, a read whose plate string already exists for the same source
    # updates that row instead of adding one. Off by default, because it also
    # merges a vehicle genuinely passing twice in quick succession, and because
    # OCR that misreads the same plate differently each time (the usual case at
    # low confidence) slips straight through it. Useful on a gate camera where
    # every duplicate matters more than every distinct pass.
    alpr_duplicate_window_seconds: float = 0.0
    # Save the full frame alongside the plate crop, with the vehicle boxed.
    # The crop proves the characters; only the frame shows what they were
    # attached to, which is what makes a read verifiable by a person. Costs one
    # extra JPEG (~150 KB at 720p) per vehicle read and one frame copy per
    # queued attempt -- turn it off if disk or CPU is tight.
    alpr_save_frame: bool = True
    # EasyOCR languages. Indonesian plates use latin characters -> "en".
    ocr_languages: str = "en"
    # Device for EasyOCR ("cpu" / "cuda" / "mps"). Empty => follow the detector
    # device, except on MPS where EasyOCR is kept on the CPU by default: its
    # recognition net is small and some of its ops fall back to the CPU anyway,
    # so sharing the GPU with YOLO usually costs more than it gains.
    ocr_device: str = ""

    # --- Face recognition (OpenCV YuNet + SFace) ---
    face_enabled: bool = True
    # ONNX model paths. Empty => default under weights_dir/face/. In Docker these
    # are prefetched and pointed at via env (YUNET_MODEL / SFACE_MODEL).
    yunet_model: str = ""
    sface_model: str = ""
    # Cosine similarity threshold for SFace (recommended ~0.363).
    face_similarity_threshold: float = 0.363
    # YuNet detector input size. NOTE: this is only the initial size handed to
    # the OpenCV backend -- both backends then size the network from the frame
    # itself (bounded by face_max_side), so it does not bound streaming cost.
    face_det_size: int = 320
    # Longest side a frame is downscaled to before face detection. This, not
    # face_det_size, is what actually bounds the streaming cost.
    face_max_side: int = 1024
    # Which implementation runs YuNet/SFace:
    #   "auto"    ONNX Runtime when installed (GPU-capable; 86.9 -> 16.0 ms
    #             on a 720p frame with 5 faces), else OpenCV
    #   "onnx"    require ONNX Runtime; disable face recognition if missing
    #   "opencv"  force cv2.FaceDetectorYN / cv2.FaceRecognizerSF
    # The two backends produce DIFFERENT embeddings (~0.93 cosine on identical
    # input), so changing this invalidates every enrolled face -- re-enroll
    # after switching. See app/detection/face.py.
    face_backend: str = "auto"
    # Log unknown (unrecognized) faces as sightings too?
    face_log_unknown: bool = False
    # Don't log the same identity on a source more often than this (seconds).
    face_sighting_cooldown_sec: int = 20

    # --- Retention ---
    retention_days: int = 7
    retention_interval_minutes: int = 60

    # --- Streaming ---
    mjpeg_fps: int = 15
    jpeg_quality: int = 70

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def plates_dir(self) -> Path:
        return self.data_dir / "plates"

    @property
    def weights_dir(self) -> Path:
        return self.data_dir / "weights"

    @property
    def faces_dir(self) -> Path:
        return self.data_dir / "faces"

    @property
    def frames_dir(self) -> Path:
        """Full-frame evidence images for plate reads."""
        return self.data_dir / "frames"

    @property
    def yolo_model_path(self) -> str:
        """Absolute path to the weights, or the bare name for auto-download.

        Accepts an absolute path, a path relative to the project root (e.g.
        ``data/weights/yolo11s.pt``, which is what the .env examples use), a
        file already sitting in the weights dir, or a plain ultralytics name
        such as ``yolo11n.pt`` that gets downloaded on first use. Resolving
        here means it does not matter which directory the app is started from.
        """
        raw = self.yolo_model
        candidate = Path(raw)
        if candidate.is_absolute():
            return raw
        for base in (Path.cwd(), BASE_DIR.parent, self.weights_dir):
            resolved = base / candidate
            if resolved.exists():
                return str(resolved)
        return raw  # ultralytics will try to download it by name

    @property
    def plate_model_path(self) -> str:
        """Absolute path to the plate-detector weights, or "" when unset.

        Resolved exactly like :attr:`yolo_model_path`. Without this the .env
        value ("data/weights/...", relative to the repo root) only worked when
        the app happened to be started from there -- and a missing plate model
        fails silently back to the heuristic ROI, so the misconfiguration would
        show up as poor ANPR accuracy rather than as an error.
        """
        raw = self.plate_model
        if not raw:
            return ""
        candidate = Path(raw)
        if candidate.is_absolute():
            return raw
        for base in (Path.cwd(), BASE_DIR.parent, self.weights_dir):
            resolved = base / candidate
            if resolved.exists():
                return str(resolved)
        return raw  # ultralytics will try to download it by name

    @property
    def yunet_model_path(self) -> str:
        return self.yunet_model or str(self.weights_dir / "face" / "yunet.onnx")

    @property
    def sface_model_path(self) -> str:
        return self.sface_model or str(self.weights_dir / "face" / "sface.onnx")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plates_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

# Ultralytics stores downloaded weights relative to the working dir by default;
# point it at our persistent weights dir so Docker volumes keep them.
os.environ.setdefault("YOLO_CONFIG_DIR", str(settings.weights_dir))

# Apple Silicon (MPS): a few ops used by torch/ultralytics have no Metal kernel.
# Allowing the CPU fallback keeps inference working instead of raising; set to
# "0" in the environment to surface such gaps instead.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# CUDA: every source is its own process with its own caching allocator, so on a
# single 8 GB card the workers are competing for one pool. Expandable segments
# let the allocator grow and give back a region instead of stranding it at the
# size of the largest block it once held, which is what fragments a long run
# with variable-sized plate crops.
#
# Windows is excluded on purpose: the allocator there does not implement it and
# warns once per worker process ("expandable_segments not supported on this
# platform"), which is pure noise. Detector.trim_memory() is what keeps memory
# in check on that platform.
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
