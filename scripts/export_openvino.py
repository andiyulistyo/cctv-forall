#!/usr/bin/env python
"""Export YOLO weights to OpenVINO IR for Intel iGPUs and x86 CPUs.

Two machines this matters for:

* **Intel Core i7 (Kaby Lake, HD Graphics)** - two CPU cores are far too few
  for real-time detection, but the Gen9 iGPU is supported by the OpenVINO GPU
  plugin. Export with ``--half`` (FP16 is the iGPU's native precision).
* **AMD Ryzen (Zen 4)** - the OpenVINO GPU plugin does not support Radeon, but
  the *CPU* plugin is vendor-neutral and much faster than torch on CPU. Export
  with ``--int8`` to use the AVX-512/VNNI integer units.

    cd backend && .venv/Scripts/python ../scripts/export_openvino.py --half
    .venv/Scripts/python ../scripts/export_openvino.py --model yolo11s.pt --int8

Then point the app at the result:

    YOLO_MODEL=data/weights/yolo11n_openvino_model
    INFERENCE_IMGSZ=640     # MUST match the --imgsz used here

The exported IR has a FIXED input size, exactly like the CoreML bundle from
scripts/export_coreml.py. Requires ``pip install openvino`` (and ``nncf`` for
--int8); both are in backend/requirements-openvino.txt.
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
    ap.add_argument("--half", action="store_true", help="export FP16 (best for an Intel iGPU)")
    ap.add_argument(
        "--int8",
        action="store_true",
        help="quantize to INT8 (best on Zen 4 / VNNI; needs nncf and downloads a "
             "small calibration set)",
    )
    ap.add_argument(
        "--data",
        default="coco128.yaml",
        help="calibration dataset for --int8 (default: coco128.yaml)",
    )
    args = ap.parse_args()

    from app.config import settings

    model_path = args.model or settings.yolo_model
    imgsz = args.imgsz or settings.inference_imgsz

    try:
        import openvino  # noqa: F401
    except ImportError:
        print(
            "openvino is required:  pip install -r backend/requirements-openvino.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.int8:
        try:
            import nncf  # noqa: F401
        except ImportError:
            print("--int8 needs nncf:  pip install nncf", file=sys.stderr)
            raise SystemExit(1)

    from ultralytics import YOLO

    print(f"Exporting {model_path} at imgsz={imgsz} (half={args.half}, int8={args.int8})...")
    kwargs = {"format": "openvino", "imgsz": imgsz, "half": args.half, "int8": args.int8}
    if args.int8:
        kwargs["data"] = args.data
    exported = Path(YOLO(model_path).export(**kwargs))

    target = settings.weights_dir / exported.name
    if exported.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(exported), str(target))

    from app.runtime import has_intel_gpu, openvino_devices

    devices = openvino_devices()
    device = "intel:gpu" if has_intel_gpu() else "intel:cpu"

    print(f"\nDone: {target}")
    print(f"OpenVINO devices on this machine: {', '.join(devices) or 'none'}")
    print("Add to your .env:")
    print(f"  YOLO_MODEL={target}")
    print(f"  INFERENCE_IMGSZ={imgsz}   # must match the exported size")
    print(f"  DEVICE={device}           # or leave empty to auto-detect the same thing")


if __name__ == "__main__":
    main()
