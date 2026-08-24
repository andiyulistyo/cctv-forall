"""Ultralytics YOLO wrapper with built-in ByteTrack tracking.

Picks the fastest device available for the weights it is given:

* ``.pt`` weights run on torch — CUDA, then Apple Silicon's Metal/MPS, then CPU.
* ``*.mlpackage`` (CoreML) runs on the Apple Neural Engine — see
  ``scripts/export_coreml.py``.
* ``*_openvino_model/`` (OpenVINO IR) runs on the Intel iGPU when one is
  present, otherwise on the OpenVINO CPU plugin — see
  ``scripts/export_openvino.py``. The CPU plugin is vendor-neutral and is the
  fast path on AMD as well, where it uses AVX-512/VNNI on Zen 4.
* ``*.engine`` (TensorRT) runs on the NVIDIA GPU it was built for — see
  ``scripts/export_tensorrt.py``. Precision is baked in at build time.
* ``*.onnx`` runs on ONNX Runtime, which picks its own provider.

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
# their own runtime (CoreML on the ANE/GPU, OpenVINO, TensorRT, ONNX Runtime,
# ...) rather than on a torch device. Grouped by runtime, because each one
# resolves to a different device and a different thread budget.
_COREML_SUFFIXES = (".mlpackage", ".mlmodel")
_TENSORRT_SUFFIXES = (".engine",)
_ONNX_SUFFIXES = (".onnx",)
_NON_TORCH_SUFFIXES = (
    _COREML_SUFFIXES + _TENSORRT_SUFFIXES + _ONNX_SUFFIXES + (".tflite", ".xml")
)

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


def _is_cuda_device(device: str) -> bool:
    """True for the device strings that mean "an NVIDIA GPU".

    ultralytics accepts a bare index ("0") as well as "cuda" / "cuda:1".
    """
    d = (device or "").strip().lower()
    return d.startswith("cuda") or d.isdigit()


def _torch_version() -> str:
    """Installed torch build, for error messages. "+cpu" is the usual culprit."""
    try:
        import torch

        return f"torch {torch.__version__}"
    except Exception:
        return "torch not installed"


def require_device(device: str) -> None:
    """Fail early, and in plain language, when a device cannot be used.

    Called at model-load time rather than from :func:`plan_inference`: planning
    has to stay pure so /api/health can report a configuration without raising,
    and so the tuning tests can run with no torch installed at all.

    Without this a CPU-only torch wheel plus ``DEVICE=cuda`` surfaces as an
    assertion from deep inside ultralytics that says nothing about the cause.
    """
    if _is_cuda_device(device) and not runtime.has_cuda():
        raise RuntimeError(
            f"DEVICE={device!r} needs a CUDA build of torch, but the installed "
            f"one ({_torch_version()}) reports no GPU. Install the matching "
            f"wheel, e.g.:\n"
            f"  pip install --force-reinstall torch torchvision "
            f"--index-url https://download.pytorch.org/whl/cu130\n"
            f"(RTX 50-series is sm_120 and needs CUDA 12.8 or newer.)"
        )
    if device.strip().lower() == "mps" and not runtime.has_mps():
        raise RuntimeError(
            "DEVICE='mps' needs Apple Silicon with a Metal-capable torch build."
        )


@dataclass(frozen=True)
class InferencePlan:
    """How a given model/device setting will actually be executed."""

    device: str      # cpu | cuda | mps | intel:cpu | intel:gpu
    backend: str     # torch | coreml | openvino | tensorrt | onnx
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
    suffix = Path(model_path).suffix.lower()
    if suffix in _TENSORRT_SUFFIXES:
        # An engine is built for one GPU and one precision; ultralytics
        # dispatches it to cuda itself. Reporting "cpu"/"coreml" here (which is
        # what the old catch-all did) made /api/health claim the GPU was idle
        # and handed the worker the wrong thread budget.
        return InferencePlan(device_setting or "cuda", "tensorrt", cpu_bound=False, half=True)
    if suffix in _ONNX_SUFFIXES:
        # ONNX Runtime picks its own execution provider; the device string only
        # tells us whether this worker still needs a full CPU thread budget.
        return InferencePlan(device, "onnx", cpu_bound=device == "cpu", half=False)
    if is_non_torch_model(model_path):
        # CoreML bundles carry their own runtime; ultralytics expects "cpu"
        # here and dispatches to the Neural Engine / GPU itself.
        return InferencePlan("cpu", "coreml", cpu_bound=False, half=False)
    # fp16 is a GPU-only win; on CPU it is emulated and slower.
    return InferencePlan(
        device, "torch", cpu_bound=device == "cpu", half=bool(half) and device != "cpu"
    )


# CPU threads a worker may use when inference is NOT on its CPU. Two is right
# for the integrated accelerators, where the CPU is also what feeds them. A
# discrete NVIDIA card is the exception: it leaves the cores genuinely free,
# but the worker still resizes, letterboxes, annotates and JPEG-encodes every
# frame on them, and 2 threads makes that the new bottleneck.
_DISCRETE_GPU_THREAD_CAP = 4
_SHARED_GPU_THREAD_CAP = 2


def gpu_thread_cap(plan: InferencePlan) -> int:
    """Thread ceiling for a worker running on ``plan``.

    Shared by the worker and by /api/health so the reported budget is the one
    actually applied.
    """
    if plan.backend == "tensorrt":
        return _DISCRETE_GPU_THREAD_CAP
    if plan.backend in ("torch", "onnx") and _is_cuda_device(plan.device):
        return _DISCRETE_GPU_THREAD_CAP
    return _SHARED_GPU_THREAD_CAP


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

    @property
    def ground_point(self) -> tuple[float, float]:
        """Where the object meets the road: bottom centre of its box.

        This, not the centroid, is what should be tested against a counting
        line. Two reasons, and both showed up in real data:

        * It is where the vehicle physically *is*. A centroid floats higher the
          taller the vehicle, so a bus and a motorcycle at the same point on the
          road cross a line at different moments.
        * It is the stable edge. ``centroid_y`` is ``(y1 + y2) / 2``, so every
          wobble of the *top* edge moves it by half as much -- and the top edge
          is the unreliable one, jumping as the model includes or excludes a
          cab, a container, a mirror. The wheels stay put.

        Measured on the sample junction feed, crossings exceeded distinct
        tracks by 18.5% for trucks, 8.5% for cars and 2.4% for motorcycles:
        ordered exactly by box size, which is the signature of box jitter
        reaching the counter.
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)

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
        require_device(self.device)
        self.model = YOLO(model_path)
        self._warmup()

    def _warmup(self) -> None:
        """Run one dummy inference so the first real frame isn't slow.

        The first MPS call compiles Metal kernels (and CoreML/OpenVINO load and
        compile the model for the target device), which can take seconds — long
        enough to stall a live stream if it happens on frame one.

        CUDA needs more than one pass: the first call pays context creation and
        the second is where cuDNN picks its algorithms for this input shape, so
        a single iteration still leaves the autotune cost on the first real
        frame.
        """
        passes = 3 if _is_cuda_device(self.device) or self.backend == "tensorrt" else 1
        try:
            blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            for _ in range(passes):
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
            # NMS across classes, not within each one. Ultralytics defaults to
            # per-class NMS, which cannot suppress two boxes that disagree about
            # what they are -- so one vehicle the model is torn about lands in
            # the results twice. Measured on a stationary taxi in the sample
            # alley: truck 0.57 [130,389,513,712] and car 0.48 [130,387,512,712]
            # -- the same object to within two pixels, kept because they carry
            # different labels. ByteTrack then tracks both, the line counts both,
            # and ANPR reads the same plate twice under two different classes.
            #
            # The overlap this suppresses is 0.7 IoU, which is far above what a
            # rider and their motorcycle share (checked on real frames: person
            # 0.89 and motorcycle 0.73 both survive), so the classes that
            # genuinely overlap here are unaffected.
            agnostic_nms=True,
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
        """Release cached GPU memory (the caching allocators grow over long runs).

        Matters most on CUDA with several workers: every source is its own
        process with its own context and its own allocator, so on an 8 GB card
        the caches are what decides whether the fourth stream starts or dies
        out of memory.
        """
        try:
            import torch

            if self.device == "mps":
                torch.mps.empty_cache()
            elif _is_cuda_device(self.device) or self.backend == "tensorrt":
                torch.cuda.empty_cache()
        except Exception:
            pass
