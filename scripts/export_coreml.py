#!/usr/bin/env python
"""Export YOLO weights to CoreML so inference can use the Apple Neural Engine.

On Apple Silicon the ANE is both faster and far more power-efficient than the
GPU for small detection models, and it leaves the GPU free for other work.
The exported bundle has a FIXED input size, so keep INFERENCE_IMGSZ equal to
the --imgsz used here.

    cd backend && .venv/bin/python ../scripts/export_coreml.py
    .venv/bin/python ../scripts/export_coreml.py --model yolo11s.pt --imgsz 640

Then point the app at the result:

    YOLO_MODEL=data/weights/yolo11n.mlpackage

Requires ``pip install coremltools``. Benchmark before switching — CoreML wins
on power draw, but the GPU (DEVICE=mps) can be faster for a single stream.
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
    ap.add_argument("--half", action="store_true", help="export fp16 (smaller, usually faster)")
    ap.add_argument("--nms", action="store_true", help="bake NMS into the model")
    args = ap.parse_args()

    from app.config import settings

    model_path = args.model or settings.yolo_model
    imgsz = args.imgsz or settings.inference_imgsz

    if sys.platform != "darwin":
        print("CoreML export only makes sense on macOS.", file=sys.stderr)
        raise SystemExit(1)
    try:
        import coremltools  # noqa: F401
    except ImportError:
        print("coremltools is required:  pip install coremltools", file=sys.stderr)
        raise SystemExit(1)

    from ultralytics import YOLO

    print(f"Exporting {model_path} at imgsz={imgsz} (half={args.half}, nms={args.nms})...")
    exported = YOLO(model_path).export(
        format="coreml", imgsz=imgsz, half=args.half, nms=args.nms
    )

    exported = Path(exported)
    target = settings.weights_dir / exported.name
    if exported.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(exported), str(target))

    print(f"\nDone: {target}")
    print("Add to your .env:")
    print(f"  YOLO_MODEL={target}")
    print(f"  INFERENCE_IMGSZ={imgsz}   # must match the exported size")


if __name__ == "__main__":
    main()
