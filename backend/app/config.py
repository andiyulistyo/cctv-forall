"""Application configuration, loaded from environment variables / .env."""
from __future__ import annotations

import os
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
    # Ultralytics weights. "yolo11n.pt" auto-downloads on first use. On Apple
    # Silicon this may also be a CoreML bundle ("yolo11n.mlpackage") exported
    # with scripts/export_coreml.py, which runs on the Neural Engine.
    yolo_model: str = "yolo11n.pt"
    # Force a device ("cpu" / "cuda" / "mps" / "0"). Empty string => auto-detect
    # (cuda -> mps -> cpu).
    device: str = ""
    # Inference image size (smaller = faster on CPU).
    inference_imgsz: int = 640
    # Detection confidence threshold.
    conf_threshold: float = 0.35
    # Process 1 out of every N frames (frame skipping). 1 = every frame.
    frame_stride: int = 2
    # Run inference in fp16. Free speed-up on MPS/CUDA, ignored on CPU.
    inference_half: bool = False
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
    # FFmpeg hardware decoder for network streams. "videotoolbox" on macOS,
    # "cuda" on NVIDIA, empty = software decoding. Falls back to software
    # automatically if the stream cannot be opened with it.
    ffmpeg_hwaccel: str = ""
    # Force RTSP over TCP (far fewer corrupt frames than the UDP default).
    rtsp_transport_tcp: bool = True
    # Cap the YouTube stream resolution. A live YouTube feed is often 1080p or
    # 4K; decoding that costs far more than the detection itself and buys no
    # accuracy at imgsz=640.
    youtube_max_height: int = 720
    # Jitter buffer for chunked sources (HLS/YouTube). They arrive in bursts of
    # a whole segment at a time; buffering this many seconds and releasing at
    # the stream's own frame rate turns the bursts back into smooth video.
    capture_buffer_seconds: float = 2.0

    # --- ANPR ---
    alpr_enabled: bool = True
    # Optional dedicated plate-detector weights. If empty, a heuristic ROI
    # (lower part of the vehicle box) is used before OCR.
    plate_model: str = ""
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
    # YuNet detector input size (smaller = faster on CPU).
    face_det_size: int = 320
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


settings = Settings()
settings.ensure_dirs()

# Ultralytics stores downloaded weights relative to the working dir by default;
# point it at our persistent weights dir so Docker volumes keep them.
os.environ.setdefault("YOLO_CONFIG_DIR", str(settings.weights_dir))

# Apple Silicon (MPS): a few ops used by torch/ultralytics have no Metal kernel.
# Allowing the CPU fallback keeps inference working instead of raising; set to
# "0" in the environment to surface such gaps instead.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
