"""Lightweight face recognition using OpenCV YuNet (detection) + SFace (embedding).

Self-contained and optional, mirroring the ANPR module (`alpr.py`): if the
ONNX models cannot be loaded, :attr:`FaceRecognizer.available` is False and the
worker simply skips face recognition. No heavy dependency is required —
``cv2.FaceDetectorYN`` and ``cv2.FaceRecognizerSF`` ship with OpenCV (>=4.5.4).

Pipeline:
- YuNet detects faces and their 5 landmarks (a per-face row of length 15:
  ``[x, y, w, h, <10 landmark coords>, score]``).
- SFace aligns each face via the landmarks and produces a 128-D embedding.
- Recognition = cosine similarity of a query embedding against an enrolled
  gallery. Above ``threshold`` (SFace cosine, default ~0.363) => recognized.
"""
from __future__ import annotations

import cv2
import numpy as np


def face_bbox(face_row: np.ndarray) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) ints from a YuNet detection row."""
    x, y, w, h = face_row[:4]
    return int(x), int(y), int(w), int(h)


class FaceRecognizer:
    def __init__(
        self,
        yunet_model: str,
        sface_model: str,
        det_size: int = 320,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_side: int = 1024,
    ):
        # Large images are downscaled to <= max_side before detection: YuNet is
        # trained for modest face scales, so very high-res photos (common at
        # enrollment) otherwise miss faces or yield false positives.
        self._max_side = max_side
        self._available = False
        self._detector = None
        self._recognizer = None
        try:
            self._detector = cv2.FaceDetectorYN.create(
                yunet_model, "", (det_size, det_size), score_threshold, nms_threshold, top_k
            )
            self._recognizer = cv2.FaceRecognizerSF.create(sface_model, "")
            self._available = True
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[Face] disabled: could not load YuNet/SFace models: {exc}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        """Return a list of YuNet detection rows (each length 15).

        Coordinates (bbox + landmarks) are always in the ORIGINAL frame space,
        even when the frame was downscaled for detection.
        """
        if not self._available:
            return []
        h, w = frame.shape[:2]
        scale = 1.0
        img = frame
        if max(h, w) > self._max_side:
            scale = self._max_side / float(max(h, w))
            img = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        sh, sw = img.shape[:2]
        self._detector.setInputSize((sw, sh))
        _, faces = self._detector.detect(img)
        if faces is None:
            return []
        out = []
        for f in faces:
            if scale != 1.0:
                f = f.copy()
                f[:14] = f[:14] / scale  # rescale bbox + landmarks to original
            out.append(f)
        return out

    def embed(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray | None:
        """Align a detected face and return its 128-D embedding."""
        if not self._available:
            return None
        try:
            aligned = self._recognizer.alignCrop(frame, face_row)
            feat = self._recognizer.feature(aligned)
            return np.asarray(feat, dtype=np.float32).flatten()
        except Exception:
            return None

    def largest_embedding(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Detect the largest face in an image and return its embedding.

        Used at enrollment time (one clear face per photo).
        """
        faces = self.detect(image_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        return self.embed(image_bgr, face)

    @staticmethod
    def match(
        feat: np.ndarray,
        gallery: np.ndarray | None,
        names: list[str],
        threshold: float,
    ) -> tuple[str | None, float]:
        """Cosine-match ``feat`` against a gallery matrix (N, 128).

        Returns ``(name, similarity)`` if best >= threshold, else
        ``(None, best_similarity)``.
        """
        if gallery is None or len(gallery) == 0:
            return None, 0.0
        fn = feat / (np.linalg.norm(feat) + 1e-9)
        gn = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-9)
        sims = gn @ fn
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        if best >= threshold:
            return names[idx], best
        return None, best
