#!/usr/bin/env python
"""Measure detection throughput for the current machine.

Runs the same YOLO model on CPU, on the Apple GPU (MPS) and — if a CoreML
bundle exists next to the weights — on the Neural Engine, so you can pick
YOLO_MODEL / DEVICE / INFERENCE_IMGSZ with numbers instead of guesses.

    cd backend && .venv/bin/python ../scripts/benchmark.py
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

from app.runtime import chip_name, describe, performance_cores  # noqa: E402


def bench(model_path: str, device: str, imgsz: int, half: bool, frames: int, size) -> float | None:
    from app.detection.detector import Detector

    try:
        det = Detector(model_path, device=device, conf=0.35, imgsz=imgsz, half=half)
    except Exception as exc:
        print(f"  ! could not load on {device}: {exc}")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="weights or .mlpackage")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=1920, help="synthetic frame width")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--devices", default="cpu,mps", help="comma separated")
    ap.add_argument("--half", action="store_true", help="also test fp16 on GPU devices")
    args = ap.parse_args()

    info = describe()
    print(f"{chip_name()} — {info['cpu_cores']} cores ({performance_cores()} performance)")
    print(f"torch {info.get('torch')}  mps={info.get('mps_available')}  cuda={info.get('cuda_available')}")
    print(f"model={args.model} imgsz={args.imgsz} input={args.width}x{args.height}\n")

    from app.detection.detector import is_non_torch_model

    size = (args.width, args.height)
    runs: list[tuple[str, bool]] = []
    if is_non_torch_model(args.model):
        # CoreML/ONNX bundles carry their own runtime; the device flag is moot.
        runs.append(("coreml", False))
    else:
        for d in [d.strip() for d in args.devices.split(",") if d.strip()]:
            runs.append((d, False))
            if args.half and d != "cpu":
                runs.append((d, True))

    results: dict[str, float] = {}
    for device, half in runs:
        label = f"{device}{' fp16' if half else ''}"
        device = "" if device == "coreml" else device
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
