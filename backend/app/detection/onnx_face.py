"""ONNX Runtime backend for YuNet (detection) and SFace (embedding).

Why this exists alongside the OpenCV DNN path in :mod:`face.py`: the pip build
of OpenCV has no CUDA backend, so ``cv2.FaceDetectorYN`` can only ever run on
the CPU -- and face recognition was by a wide margin the most expensive step in
the pipeline, roughly 8x the cost of vehicle detection.

One 1280x720 frame with 5 faces, detect + embed, on an RTX 5070
Laptop / Ryzen 9 8940HX:

    cv2.dnn (OpenCV)        86.9 ms
    onnxruntime CPU         61.8 ms   (1.4x -- the runtime alone)
    onnxruntime CUDA        16.0 ms   (5.4x)

So the GPU, not the runtime, is what carries this: ~3.9x of the 5.4x comes from
moving off the CPU. The runtime swap is still worth it on a machine without a
GPU, just far less dramatically.

Detection dominates a frame with few faces; SFace dominates one with many,
since it runs once per face and gains ~5.9x on the GPU (6.35 ms -> 1.07 ms).

The models are the stock OpenCV Zoo exports already shipped in
``data/weights/face/``; nothing needs re-exporting. Their ONNX graphs declare a
fixed 640x640 input, so the input dimensions are rewritten to symbolic at load
time and the network is fed at the same resolution the OpenCV path used. That
is what keeps detection sensitivity -- and therefore the embeddings -- equal to
the previous implementation instead of quietly losing small, distant faces.

This module produces the **same 15-column detection rows** as
``cv2.FaceDetectorYN.detect`` (``[x, y, w, h, 10 landmark coords, score]``), so
it is a drop-in for the rest of the app and the two can be compared directly.
``scripts/validate_onnx_face.py`` does exactly that and is the guard on the
hand-written anchor decoding below.
"""
from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np

# The three detection heads, and the anchor stride each is decoded with. The
# network is fully convolutional, so the grid for each head is simply the input
# size divided by its stride -- which is also why the input must be a multiple
# of the largest stride.
_STRIDES = (8, 16, 32)
_SIZE_MULTIPLE = 32
# SFace consumes the aligned 112x112 crop that cv2's alignCrop produces.
_SFACE_SIZE = 112


def _add_torch_dll_dir() -> None:
    """Let ONNX Runtime find the CUDA libraries that torch already ships.

    onnxruntime-gpu 1.29 links against CUDA 13 (``cublasLt64_13.dll``) and
    cuDNN 9 -- exactly what the ``+cu130`` torch wheel unpacks into
    ``torch/lib``. Pointing Windows at that directory means the GPU path works
    off the torch install alone, with no separate CUDA Toolkit to install and
    no second copy of CUDA to keep in sync.
    """
    if os.name != "nt":
        return
    try:
        import torch

        lib = pathlib.Path(torch.__file__).parent / "lib"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
    except Exception:
        pass


def available_providers(prefer_gpu: bool = True) -> list[str]:
    """Execution providers to try, best first.

    CPU is always appended. ONNX Runtime falls back to it per-operator anyway,
    and its CPU path is still ~1.4x faster than the OpenCV one, so a machine
    without a usable GPU ends up better off than before rather than worse.
    """
    import onnxruntime as ort

    providers: list[str] = []
    if prefer_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _load_dynamic(model_path: str) -> bytes:
    """Return the model with its spatial input dimensions made symbolic.

    The shipped YuNet export pins 1x3x640x640. Feeding every frame at 640x640
    would halve the working resolution of a 720p camera compared with the
    OpenCV path (which reshapes the net per frame), and small or distant faces
    would stop being detected. Rewriting the two spatial dims to symbolic gives
    back that freedom without re-exporting or shipping a second model file.
    """
    import onnx

    model = onnx.load(model_path)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    dims[2].Clear()
    dims[2].dim_param = "height"
    dims[3].Clear()
    dims[3].dim_param = "width"
    # The recorded output shapes were derived from the fixed input; leaving
    # them in place makes ORT reject anything else as a shape mismatch.
    for out in model.graph.output:
        for dim in out.type.tensor_type.shape.dim:
            dim.Clear()
            dim.dim_param = "?"
    return model.SerializeToString()


class OnnxFaceBackend:
    """YuNet + SFace on ONNX Runtime, with the cv2-compatible row format."""

    def __init__(
        self,
        yunet_model: str,
        sface_model: str,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_side: int = 1024,
        prefer_gpu: bool = True,
    ):
        _add_torch_dll_dir()
        import onnxruntime as ort

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        # Same default as the OpenCV path: YuNet is trained for modest face
        # scales, so very large images are downscaled before detection.
        self._max_side = max_side

        opts = ort.SessionOptions()
        # ORT is noisy about falling back to CPU for individual operators; the
        # provider actually in use is reported by self.provider instead.
        opts.log_severity_level = 3
        providers = available_providers(prefer_gpu)
        self._det = ort.InferenceSession(_load_dynamic(yunet_model), opts, providers=providers)
        self._rec = ort.InferenceSession(sface_model, opts, providers=providers)
        self._det_input = self._det.get_inputs()[0].name
        self._rec_input = self._rec.get_inputs()[0].name
        # Index outputs by name so a different export of the same model cannot
        # silently reorder the heads.
        self._det_outputs = [o.name for o in self._det.get_outputs()]
        self.provider = self._det.get_providers()[0]

        # cv2's SFace wrapper is still the cheapest correct way to do the
        # landmark-based similarity transform; only the network moves to ORT.
        self._aligner = cv2.FaceRecognizerSF.create(sface_model, "")

        # Anchor grids depend only on the input size, and a given camera keeps
        # the same one for its whole run, so build them once per size.
        self._anchor_cache: dict[tuple[int, int], dict] = {}

    def _anchors(self, h: int, w: int) -> dict:
        key = (h, w)
        cached = self._anchor_cache.get(key)
        if cached is None:
            cached = {}
            for stride in _STRIDES:
                gh, gw = h // stride, w // stride
                cols, rows = np.meshgrid(np.arange(gw), np.arange(gh))
                cached[stride] = (cols.reshape(-1).astype(np.float32),
                                  rows.reshape(-1).astype(np.float32))
            self._anchor_cache[key] = cached
        return cached

    # ---------------- detection ----------------

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Downscale to ``max_side`` and pad to a multiple of 32.

        Padding goes on the right and bottom only, so mapping a detection back
        to the original frame is a single division by ``scale`` -- there is no
        offset to subtract.
        """
        h, w = frame.shape[:2]
        scale = min(1.0, self._max_side / float(max(h, w)))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = frame if scale == 1.0 else cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # Actual scale after integer rounding, so boxes map back exactly.
        scale = nw / float(w)

        ph = (nh + _SIZE_MULTIPLE - 1) // _SIZE_MULTIPLE * _SIZE_MULTIPLE
        pw = (nw + _SIZE_MULTIPLE - 1) // _SIZE_MULTIPLE * _SIZE_MULTIPLE
        if (ph, pw) != (nh, nw):
            canvas = np.zeros((ph, pw, 3), dtype=img.dtype)
            canvas[:nh, :nw] = img
            img = canvas
        return img, scale

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        """Detect faces, returning cv2-compatible rows in ORIGINAL frame coords."""
        if frame is None or frame.size == 0:
            return []
        img, scale = self._preprocess(frame)
        h, w = img.shape[:2]
        # YuNet takes raw BGR values: no mean subtraction, no 1/255 scaling --
        # the same blob cv2.dnn builds for it.
        blob = img.transpose(2, 0, 1)[None].astype(np.float32)
        outs = dict(zip(self._det_outputs, self._det.run(None, {self._det_input: blob})))
        anchors = self._anchors(h, w)

        boxes, scores, kps = [], [], []
        for stride in _STRIDES:
            cls = outs[f"cls_{stride}"][0].reshape(-1)
            obj = outs[f"obj_{stride}"][0].reshape(-1)
            bbox = outs[f"bbox_{stride}"][0].reshape(-1, 4)
            kp = outs[f"kps_{stride}"][0].reshape(-1, 10)
            cols, rows = anchors[stride]

            # YuNet scores a candidate as the geometric mean of its
            # classification and objectness heads.
            score = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
            keep = score >= self.score_threshold
            if not np.any(keep):
                continue

            c, r = cols[keep], rows[keep]
            b, k, s = bbox[keep], kp[keep], score[keep]
            # Centre is an offset from the anchor cell; size is log-encoded.
            cx = (c + b[:, 0]) * stride
            cy = (r + b[:, 1]) * stride
            bw = np.exp(b[:, 2]) * stride
            bh = np.exp(b[:, 3]) * stride

            landmarks = np.empty((len(s), 10), dtype=np.float32)
            landmarks[:, 0::2] = (c[:, None] + k[:, 0::2]) * stride
            landmarks[:, 1::2] = (r[:, None] + k[:, 1::2]) * stride

            boxes.append(np.stack([cx - bw / 2.0, cy - bh / 2.0, bw, bh], axis=1))
            scores.append(s)
            kps.append(landmarks)

        if not boxes:
            return []
        boxes = np.concatenate(boxes).astype(np.float32)
        scores = np.concatenate(scores).astype(np.float32)
        kps = np.concatenate(kps).astype(np.float32)

        idxs = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.score_threshold, self.nms_threshold,
            top_k=self.top_k,
        )
        if len(idxs) == 0:
            return []
        idxs = np.asarray(idxs).reshape(-1)
        # Strongest first, matching what cv2.FaceDetectorYN returns.
        idxs = idxs[np.argsort(-scores[idxs])]

        out_rows: list[np.ndarray] = []
        for i in idxs:
            row = np.empty(15, dtype=np.float32)
            row[:4] = boxes[i] / scale
            row[4:14] = kps[i] / scale
            row[14] = scores[i]
            out_rows.append(row)
        return out_rows

    # ---------------- embedding ----------------

    def embed(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray | None:
        """Align a detected face and return its 128-D SFace embedding."""
        try:
            aligned = self._aligner.alignCrop(frame, face_row)
            if aligned is None or aligned.size == 0:
                return None
            if aligned.shape[0] != _SFACE_SIZE or aligned.shape[1] != _SFACE_SIZE:
                aligned = cv2.resize(aligned, (_SFACE_SIZE, _SFACE_SIZE))
            blob = aligned.transpose(2, 0, 1)[None].astype(np.float32)
            feat = self._rec.run(None, {self._rec_input: blob})[0]
            return np.asarray(feat, dtype=np.float32).flatten()
        except Exception:
            return None
