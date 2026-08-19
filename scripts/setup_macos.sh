#!/usr/bin/env bash
# One-time setup for running the dashboard natively on Apple Silicon
# (Mac mini M4 / M4 Pro, MacBook M-series, ...).
#
# Docker Desktop on macOS cannot pass the GPU or the Neural Engine into a
# container, so a native install is the only way to get hardware acceleration.
#
#   ./scripts/setup_macos.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$ROOT/backend"
VENV="$BACKEND/.venv"

echo "==> Checking the machine"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is for macOS. Use Docker on other platforms." >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "warning: not Apple Silicon — the app will run, but on the CPU only." >&2
fi
sysctl -n machdep.cpu.brand_string || true

echo "==> Checking prerequisites"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install it with:  brew install ffmpeg" >&2
  echo "(OpenCV ships its own FFmpeg for decoding, but the CLI is handy for testing.)" >&2
fi

# Prefer a Python that has wheels for every dependency.
PYTHON=""
for candidate in python3.12 python3.11 python3.13 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$(command -v "$candidate")"; break; fi
done
if [[ -z "$PYTHON" ]]; then
  echo "No python3 found. Install it with:  brew install python@3.12" >&2
  exit 1
fi
echo "Using $PYTHON ($("$PYTHON" -V))"

echo "==> Creating the virtualenv at backend/.venv"
[[ -d "$VENV" ]] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel >/dev/null

echo "==> Installing PyTorch (Apple Silicon build, includes Metal/MPS support)"
"$VENV/bin/python" -m pip install torch torchvision

echo "==> Installing the backend requirements"
"$VENV/bin/python" -m pip install -r "$BACKEND/requirements.txt"

echo "==> Fetching model weights into data/weights"
WEIGHTS="$ROOT/data/weights"
mkdir -p "$WEIGHTS/face"
YOLO_MODEL_NAME="${YOLO_MODEL_NAME:-yolo11s.pt}"   # M4 Pro handles "s" easily
if [[ ! -f "$WEIGHTS/$YOLO_MODEL_NAME" ]]; then
  ( cd "$WEIGHTS" && YOLO_CONFIG_DIR="$WEIGHTS" "$VENV/bin/python" -c \
      "from ultralytics import YOLO; YOLO('$YOLO_MODEL_NAME')" )
fi
[[ -f "$WEIGHTS/face/yunet.onnx" ]] || curl -sSL -o "$WEIGHTS/face/yunet.onnx" \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
[[ -f "$WEIGHTS/face/sface.onnx" ]] || curl -sSL -o "$WEIGHTS/face/sface.onnx" \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

echo "==> Warming up EasyOCR (downloads its detection/recognition models once)"
"$VENV/bin/python" -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)" >/dev/null

echo "==> Building the frontend"
if command -v npm >/dev/null 2>&1; then
  ( cd "$ROOT/frontend" && npm install && npm run build )
else
  echo "npm not found — skipping the frontend build. Install Node (brew install node)" >&2
  echo "and run: cd frontend && npm install && npm run build" >&2
fi

echo "==> Creating .env"
if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.macos.example" "$ROOT/.env"
  echo "Wrote .env from .env.macos.example — change ADMIN_PASSWORD and JWT_SECRET."
else
  echo ".env already exists, leaving it alone."
  echo "Compare it against .env.macos.example for the Apple Silicon settings."
fi

cat <<EOF

Setup complete.

  Start the app:      ./scripts/run_macos.sh
  Measure throughput: backend/.venv/bin/python scripts/benchmark.py \\
                        --model data/weights/$YOLO_MODEL_NAME --half
  Neural Engine:      backend/.venv/bin/pip install coremltools
                      backend/.venv/bin/python scripts/export_coreml.py

EOF
