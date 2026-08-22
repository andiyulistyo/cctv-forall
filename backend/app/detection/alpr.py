"""Automatic License-Plate Recognition (ANPR) for Indonesian plates.

This module is intentionally self-contained and optional: if EasyOCR (or the
model weights) cannot be loaded, :meth:`ALPR.available` is False and the worker
simply skips plate reading. OCR is done with EasyOCR (latin characters, which
covers Indonesian plates). An optional dedicated plate-detector (YOLO) can be
plugged in via ``plate_model``; otherwise a heuristic region of interest (the
lower-centre of the vehicle box) is used before OCR.

Indonesian plate format (roughly): 1-2 area letters, 1-4 digits, 1-3 suffix
letters, e.g. "B 1234 XYZ" / "AD 12 AB".
"""
from __future__ import annotations

import re

import cv2
import numpy as np

# Loose Indonesian plate pattern after removing spaces.
_PLATE_RE = re.compile(r"^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$")


def normalize_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def looks_like_plate(normalized: str) -> bool:
    if not (4 <= len(normalized) <= 9):
        return False
    return bool(_PLATE_RE.match(normalized))


class ALPR:
    def __init__(
        self,
        languages: str = "en",
        device: str = "cpu",
        plate_model: str = "",
        plate_imgsz: int = 320,
        plate_conf: float = 0.25,
        min_confidence: float = 0.20,
        half: bool = False,
    ):
        self._reader = None
        self._plate_detector = None
        self._available = False
        self.device = device or "cpu"
        self.plate_imgsz = plate_imgsz
        self.plate_conf = plate_conf
        self.min_confidence = min_confidence
        # fp16 is a GPU-only win, exactly as for the vehicle detector.
        self.half = bool(half) and self.device != "cpu"
        try:
            import easyocr

            langs = [s.strip() for s in languages.split(",") if s.strip()] or ["en"]
            # EasyOCR accepts gpu=False, gpu=True or an explicit device string.
            # Passing the string keeps us in control (e.g. "mps" on Apple
            # Silicon) instead of relying on its own auto-detection.
            gpu_arg = False if self.device == "cpu" else self.device
            try:
                self._reader = easyocr.Reader(langs, gpu=gpu_arg, verbose=False)
            except Exception as exc:
                # Some EasyOCR builds choke on non-CUDA accelerators; the CPU
                # path always works and OCR is only run on small plate crops.
                print(f"[ALPR] {self.device} unavailable ({exc}); falling back to CPU")
                self.device = "cpu"
                self._reader = easyocr.Reader(langs, gpu=False, verbose=False)
            self._available = True
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[ALPR] disabled: could not init EasyOCR: {exc}")
            self._available = False

        if plate_model:
            try:
                from ultralytics import YOLO

                self._plate_detector = YOLO(plate_model)
                # Warm it on the target device: the first call on CUDA pays
                # context + autotune, and here that would land on a live frame.
                self._plate_detector.predict(
                    np.zeros((plate_imgsz, plate_imgsz, 3), dtype=np.uint8),
                    imgsz=self.plate_imgsz,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )
            except Exception as exc:
                print(f"[ALPR] plate detector disabled: {exc}")
                self._plate_detector = None

    @property
    def available(self) -> bool:
        return self._available

    def _plate_roi(self, vehicle_crop: np.ndarray) -> np.ndarray:
        """Return the region most likely to contain the plate."""
        if self._plate_detector is not None:
            # Without an explicit device this silently ran on the ultralytics
            # default rather than alongside the vehicle detector.
            res = self._plate_detector.predict(
                vehicle_crop,
                imgsz=self.plate_imgsz,
                device=self.device,
                half=self.half,
                conf=self.plate_conf,
                verbose=False,
            )
            if res and res[0].boxes is not None and len(res[0].boxes) > 0:
                # Highest-confidence plate box.
                boxes = res[0].boxes
                idx = int(boxes.conf.cpu().numpy().argmax())
                x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[idx].astype(int)
                roi = vehicle_crop[max(0, y1):y2, max(0, x1):x2]
                if roi.size:
                    return roi
        # Heuristic: plates sit low on the vehicle; take the bottom 45%.
        h = vehicle_crop.shape[0]
        return vehicle_crop[int(h * 0.55):, :]

    def read_plate(self, vehicle_crop: np.ndarray) -> tuple[str, float, np.ndarray] | None:
        """Try to read a plate from a vehicle crop.

        Returns ``(plate_text, confidence, plate_image)`` or None.
        """
        if not self._available or vehicle_crop is None or vehicle_crop.size == 0:
            return None

        roi = self._plate_roi(vehicle_crop)
        if roi.size == 0:
            return None

        # Upscale small ROIs to help OCR.
        if roi.shape[1] < 200:
            scale = 200 / max(1, roi.shape[1])
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        try:
            results = self._reader.readtext(roi)
        except Exception:
            return None

        best: tuple[str, float] | None = None
        for _bbox, text, conf in results:
            norm = normalize_plate(text)
            if not norm:
                continue
            score = conf + (0.5 if looks_like_plate(norm) else 0.0)
            if best is None or score > best[1]:
                best = (norm, float(conf))

        if best is None:
            return None
        text, conf = best
        # Require a plausible plate to cut down on noise.
        if not looks_like_plate(text):
            return None
        # ...and require the OCR to have actually been sure of it. The pattern
        # check alone is weak: a blurry plate at distance still yields a string
        # that matches it, just not the right one. Returning None here leaves
        # the track unresolved so a closer frame gets another go.
        if conf < self.min_confidence:
            return None
        return text, conf, roi
