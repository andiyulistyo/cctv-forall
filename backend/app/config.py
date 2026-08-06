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
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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
    # Ultralytics weights. "yolo11n.pt" auto-downloads on first use.
    yolo_model: str = "yolo11n.pt"
    # Force a device ("cpu" / "cuda" / "0"). Empty string => auto-detect.
    device: str = ""
    # Inference image size (smaller = faster on CPU).
    inference_imgsz: int = 640
    # Detection confidence threshold.
    conf_threshold: float = 0.35
    # Process 1 out of every N frames (frame skipping). 1 = every frame.
    frame_stride: int = 2

    # --- ANPR ---
    alpr_enabled: bool = True
    # Optional dedicated plate-detector weights. If empty, a heuristic ROI
    # (lower part of the vehicle box) is used before OCR.
    plate_model: str = ""
    # EasyOCR languages. Indonesian plates use latin characters -> "en".
    ocr_languages: str = "en"

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

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plates_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

# Ultralytics stores downloaded weights relative to the working dir by default;
# point it at our persistent weights dir so Docker volumes keep them.
os.environ.setdefault("YOLO_CONFIG_DIR", str(settings.weights_dir))
