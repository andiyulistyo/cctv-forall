"""Ultralytics YOLO wrapper with built-in ByteTrack tracking.

Picks the fastest device available for the weights it is given:

* ``.pt`` weights run on torch — CUDA, then Apple Silicon's Metal/MPS, then CPU.
* ``*.mlpackage`` (CoreML) runs on the Apple Neural Engine — see
  ``scripts/export_coreml.py``.
* ``*_openvino_model/`` (OpenVINO IR) runs on the Intel iGPU when one is
  present, otherwise on the OpenVINO CPU plugin — see
  ``scripts/export_openvino.py``. The CPU plugin is vendor-neutral and is the
  fast path on AMD as well, where it uses AVX-512/VNNI on Zen 4.

Everything else in the app asks :func:`plan_inference` rather than working this
out again: the worker needs it for its thread budget and /api/health reports it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import runtime

# COCO class id -> our friendly class name (only the ones we care about).
COCO_ID_TO_NAME = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
NAME_TO_COCO_ID = {v: k for k, v in COCO_ID_TO_NAME.items()}

# Model formats that are not plain PyTorch weights and therefore run through
# their own runtime (CoreML on the ANE/GPU, OpenVINO, ONNX Runtime, ...) rather
# than on a torch device.
_NON_TORCH_SUFFIXES = (".mlpackage", ".mlmodel", ".onnx", ".engine", ".tflite", ".xml")

# Ultralytics writes its OpenVINO IR into a directory with this suffix.
_OPENVINO_DIR_SUFFIX = "_openvino_model"


def is_openvino_model(model_path: str) -> bool:
    """True for an ultralytics OpenVINO export (a directory) or a raw IR .xml.

    Unlike the other formats this one is a *directory*, so a suffix check alone
    would miss it.
    """
    p = Path(model_path)
    return p.name.endswith(_OPENVINO_DIR_SUFFIX) or p.suffix.lower() == ".xml"


def is_non_torch_model(model_path: str) -> bool:
    return (
        is_openvino_model(model_path)
        or Path(model_path).suffix.lower() in _NON_TORCH_SUFFIXES
    )


def pick_device(preferred: str = "", model_path: str = "") -> str:
    """Resolve the inference device: explicit setting, else best available."""
    if preferred:
        return preferred
    if model_path and is_openvino_model(model_path):
        # Resolve explicitly instead of leaving it to OpenVINO's AUTO plugin:
        # the worker sizes its CPU thread budget from this answer, and "AUTO"
        # would not tell it whether the CPU is doing the work.
        return "intel:gpu" if runtime.has_intel_gpu() else "intel:cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon GPU (Metal Performance Shaders).
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass(frozen=True)
class InferencePlan:
    """How a given model/device setting will actually be executed."""

    device: str      # cpu | cuda | mps | intel:cpu | intel:gpu
    backend: str     # torch | coreml | openvino
    cpu_bound: bool  # True => this worker needs a full CPU thread budget
    half: bool


def plan_inference(model_path: str, device_setting: str = "", half: bool = False) -> InferencePlan:
    """Single source of truth for device/backend selection.

    Used by :class:`Detector`, by the worker (to size its thread budget) and by
    /api/health, so all three always agree.
    """
    device = pick_device(device_setting, model_path)
    if is_openvino_model(model_path):
        # Precision is baked in at export time; the half flag is meaningless.
        on_gpu = device.startswith("intel:") and device.split(":", 1)[1].lower() != "cpu"
        return InferencePlan(device, "openvino", cpu_bound=not on_gpu, half=False)
    if is_non_torch_model(model_path):
        # CoreML/ONNX bundles carry their own runtime; ultralytics expects
        # "cpu" here and dispatches to the Neural Engine / GPU itself.
        return InferencePlan("cpu", "coreml", cpu_bound=False, half=False)
    # fp16 is a GPU-only win; on CPU it is emulated and slower.
    return InferencePlan(
        device, "torch", cpu_bound=device == "cpu", half=bool(half) and device != "cpu"
    )


@dataclass
class Detection:
    track_id: int
    class_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def scaled(self, factor: float) -> "Detection":
        """Same detection expressed in a frame scaled by ``factor``."""
        if factor == 1.0:
            return self
        return Detection(
            self.track_id,
            self.class_name,
            self.conf,
            self.x1 * factor,
            self.y1 * factor,
            self.x2 * factor,
            self.y2 * factor,
        )


class Detector:
    def __init__(
        self,
        model_path: str,
        device: str = "",
        conf: float = 0.35,
        imgsz: int = 640,
        half: bool = False,
    ):
        from ultralytics import YOLO

        self.conf = conf
        self.imgsz = imgsz
        self.plan = plan_inference(model_path, device, half)
        self.device = self.plan.device
        self.half = self.plan.half
        self.backend = self.plan.backend
        self.model = YOLO(model_path)
        self._warmup()

    def _warmup(self) -> None:
        """Run one dummy inference so the first real frame isn't slow.

        The first MPS call compiles Metal kernels (and CoreML/OpenVINO load and
        compile the model for the target device), which can take seconds — long
        enough to stall a live stream if it happens on frame one.
        """
        try:
            blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                blank,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )
        except Exception as exc:  # non-fatal: real inference may still work
            print(f"[Detector] warmup skipped: {exc}")

    def track(self, frame, enabled_classes: list[str]) -> list[Detection]:
        """Run detection + tracking on a frame, filtered to enabled classes."""
        class_ids = [NAME_TO_COCO_ID[c] for c in enabled_classes if c in NAME_TO_COCO_ID]
        if not class_ids:
            return []

        results = self.model.track(
            frame,
            persist=True,
            classes=class_ids,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        if not results:
            return []

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.id is None:
            return []

        detections: list[Detection] = []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), tid, cid, cf in zip(xyxy, ids, clss, confs):
            name = COCO_ID_TO_NAME.get(int(cid))
            if name is None:
                continue
            detections.append(
                Detection(int(tid), name, float(cf), float(x1), float(y1), float(x2), float(y2))
            )
        return detections

    def trim_memory(self) -> None:
        """Release cached GPU memory (MPS grows its cache over long runs)."""
        if self.device != "mps":
            return
        try:
            import torch

            torch.mps.empty_cache()
        except Exception:
            pass
