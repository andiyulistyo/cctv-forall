#!/usr/bin/env python
"""Export YOLO weights to a TensorRT engine for an NVIDIA GPU.

This is an *optional* last step. Running on the GPU does not need it: point
YOLO_MODEL at the .pt weights, set DEVICE=cuda and INFERENCE_HALF=true, and
torch already uses the card. See .env.nvidia.example.

Measure before you commit to it. On an RTX 5070 Laptop with yolo11m at
imgsz=640, roughly 4 of the ~11 ms spent per frame is the forward pass; the
rest is the letterbox, NMS and ultralytics' own Python overhead. TensorRT only
compresses that 4 ms, so the end-to-end gain is much smaller than the
inference-only speedups usually quoted for it.

    cd backend && .venv/Scripts/python ../scripts/export_tensorrt.py --imgsz 640
    .venv/Scripts/python ../scripts/export_tensorrt.py --model ../data/weights/yolo11m.pt

Then point the app at the result:

    YOLO_MODEL=data/weights/yolo11m.engine
    INFERENCE_IMGSZ=640     # MUST match the --imgsz used here

Three things to know about the engine it produces:

* It is tied to this GPU, this driver and this TensorRT version. Change any of
  them -- including a routine driver update -- and it must be rebuilt. Keep the
  .pt weights around; they are the portable copy.
* Its input size is FIXED, exactly like the OpenVINO IR from
  scripts/export_openvino.py, so INFERENCE_IMGSZ must equal --imgsz.
* Precision is baked in at build time, which is why the app reports half=True
  for an engine and ignores INFERENCE_HALF.

Requires ``pip install -r backend/requirements-cuda.txt``. The build takes
several minutes and is largely silent while TensorRT profiles kernels.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="weights path (default: settings.yolo_model)")
    ap.add_argument("--imgsz", type=int, default=0, help="default: settings.inference_imgsz")
    ap.add_argument(
        "--fp32",
        action="store_true",
        help="build without fp16 (slower; only useful to isolate a precision problem)",
    )
    ap.add_argument(
        "--int8",
        action="store_true",
        help="quantize to INT8 (fastest, needs a calibration set and can cost accuracy)",
    )
    ap.add_argument(
        "--data",
        default="coco128.yaml",
        help="calibration dataset for --int8 (default: coco128.yaml)",
    )
    ap.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="build-time workspace in GiB (default: 4)",
    )
    ap.add_argument("--device", default="0", help="GPU index to build for (default: 0)")
    args = ap.parse_args()

    from app.config import settings
    from app.runtime import cuda_devices

    model_path = args.model or settings.yolo_model
    imgsz = args.imgsz or settings.inference_imgsz

    gpus = cuda_devices()
    if not gpus:
        print(
            "No CUDA device available. TensorRT builds an engine for the GPU it "
            "runs on, so this must be run on the target machine with a CUDA "
            "build of torch installed:\n"
            "  pip install --force-reinstall torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu130",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print(
            "tensorrt is required:  pip install -r backend/requirements-cuda.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)

    from ultralytics import YOLO

    half = not args.fp32 and not args.int8
    for gpu in gpus:
        print(f"GPU {gpu['index']}: {gpu['name']} — sm_{gpu['capability'].replace('.', '')}")
    print(
        f"Exporting {model_path} at imgsz={imgsz} "
        f"(half={half}, int8={args.int8}) — this takes a few minutes..."
    )

    kwargs = {
        "format": "engine",
        "imgsz": imgsz,
        "half": half,
        "int8": args.int8,
        "device": args.device,
        "workspace": args.workspace,
    }
    if args.int8:
        kwargs["data"] = args.data
    exported = Path(YOLO(model_path).export(**kwargs))

    target = settings.weights_dir / exported.name
    if exported.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        shutil.move(str(exported), str(target))

    print(f"\nDone: {target}")
    print("Add to your .env:")
    print(f"  YOLO_MODEL={target}")
    print(f"  INFERENCE_IMGSZ={imgsz}   # must match the exported size")
    print("  DEVICE=cuda")
    print("\nCompare it against the .pt weights before switching:")
    print(f"  python scripts/benchmark.py --model {target} --imgsz {imgsz}")
    print(f"  python scripts/benchmark.py --model {model_path} --imgsz {imgsz}")


if __name__ == "__main__":
    main()
