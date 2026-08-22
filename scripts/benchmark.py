#!/usr/bin/env python
"""Measure detection throughput for the current machine.

Runs the same model on every backend this machine can actually use — torch on
CPU/CUDA/MPS, the Apple Neural Engine via a CoreML bundle, the Intel iGPU or
the OpenVINO CPU plugin via an OpenVINO IR — so you can pick YOLO_MODEL /
DEVICE / INFERENCE_IMGSZ with numbers instead of guesses.

    cd backend && .venv/bin/python ../scripts/benchmark.py
    .venv/Scripts/python ../scripts/benchmark.py --model ../data/weights/yolo11n_openvino_model
    .venv/bin/python ../scripts/benchmark.py --model yolo11s.pt --imgsz 640 --frames 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402

from app.runtime import (  # noqa: E402
    chip_name,
    cuda_devices,
    describe,
    has_cuda,
    has_intel_gpu,
    has_mps,
    openvino_devices,
    physical_cores,
)


def bench(model_path: str, device: str, imgsz: int, half: bool, frames: int, size) -> float | None:
    from app.detection.detector import Detector

    try:
        det = Detector(model_path, device=device, conf=0.35, imgsz=imgsz, half=half)
    except Exception as exc:
        print(f"  ! could not load on {device or 'auto'}: {exc}")
        return None

    rng = np.random.default_rng(0)
    # Random noise is a worst case for NMS but keeps timing honest and offline.
    clip = [rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8) for _ in range(8)]
    classes = ["person", "car", "motorcycle", "truck", "bus"]

    for f in clip[:2]:  # settle
        det.track(f, classes)

    start = time.perf_counter()
    for i in range(frames):
        det.track(clip[i % len(clip)], classes)
    elapsed = time.perf_counter() - start
    return frames / elapsed


def auto_devices(model_path: str) -> list[str]:
    """Devices worth trying for this model on this machine.

    An OpenVINO IR only ever runs on OpenVINO devices, a CoreML bundle only on
    the Apple runtime; torch weights can go to whatever torch found.
    """
    from app.detection.detector import is_non_torch_model, is_openvino_model

    from pathlib import Path as _Path

    if is_openvino_model(model_path):
        devices = ["intel:cpu"]
        if has_intel_gpu():
            devices.append("intel:gpu")
        return devices
    if _Path(model_path).suffix.lower() == ".engine":
        # A TensorRT engine only runs on the GPU it was built for. Listing it
        # here lets you compare an engine against .pt fp16 in one run.
        return ["cuda"]
    if is_non_torch_model(model_path):
        return [""]  # the bundle picks its own runtime
    devices = ["cpu"]
    if has_cuda():
        devices.append("cuda")
    if has_mps():
        devices.append("mps")
    return devices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="weights, .mlpackage or _openvino_model/")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=1920, help="synthetic frame width")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument(
        "--devices",
        default="auto",
        help="comma separated, or 'auto' to test everything this machine supports",
    )
    ap.add_argument(
        "--half",
        action="store_true",
        help="also test fp16 on MPS (CUDA is measured in fp16 by default)",
    )
    ap.add_argument(
        "--no-half",
        action="store_true",
        help="skip the fp16 run on CUDA and measure fp32 only",
    )
    args = ap.parse_args()

    info = describe()
    print(f"{chip_name()} — {info['cpu_cores']} logical / {physical_cores()} physical cores")
    print(
        f"torch {info.get('torch')}  mps={info.get('mps_available')}  "
        f"cuda={info.get('cuda_available')} (CUDA {info.get('cuda_version')})  "
        f"openvino={info.get('openvino')} {list(openvino_devices())}"
    )
    for gpu in cuda_devices():
        # Compute capability is the number that decides whether the installed
        # wheel has kernels for this card at all (sm_120 = Blackwell).
        print(
            f"GPU {gpu['index']}: {gpu['name']} — {gpu['total_memory_mb']} MiB, "
            f"compute capability {gpu['capability']}"
        )
    print(f"model={args.model} imgsz={args.imgsz} input={args.width}x{args.height}\n")

    if args.devices.strip().lower() == "auto":
        devices = auto_devices(args.model)
    else:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    size = (args.width, args.height)
    runs: list[tuple[str, bool]] = []
    for d in devices:
        # fp16 is the production setting on CUDA (.env.nvidia.example sets
        # INFERENCE_HALF=true), so measure it by default rather than hiding the
        # real number behind a flag. MPS stays opt-in: results there are mixed.
        cuda_half = d == "cuda" and not args.no_half
        if not cuda_half:
            runs.append((d, False))
        if cuda_half or (args.half and d in ("cuda", "mps")):
            runs.append((d, True))

    results: dict[str, float] = {}
    for device, half in runs:
        label = f"{device or 'auto'}{' fp16' if half else ''}"
        print(f"[{label}] running {args.frames} frames...")
        fps = bench(args.model, device, args.imgsz, half, args.frames, size)
        if fps:
            results[label] = fps
            print(f"  {fps:6.1f} fps  ({1000 / fps:5.1f} ms/frame)")

    if results:
        print("\nsummary (single stream, detection only):")
        for label, fps in sorted(results.items(), key=lambda kv: -kv[1]):
            print(f"  {label:10s} {fps:6.1f} fps")
        best = max(results.values())
        print(
            f"\nWith FRAME_STRIDE=2 the best device sustains roughly "
            f"{best * 2 / 25:.1f} concurrent 25 fps streams before detection "
            f"becomes the bottleneck (decoding costs extra)."
        )


if __name__ == "__main__":
    main()
