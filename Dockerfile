# syntax=docker/dockerfile:1

# ---------- Stage 1: build the React frontend ----------
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build   # outputs /frontend/dist

# ---------- Stage 2: python backend (CPU) ----------
FROM python:3.11-slim AS backend

# System deps: ffmpeg for OpenCV stream decoding; libs for OpenCV/torch runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU build of torch/torchvision (kept out of requirements.txt on purpose).
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights so containers work offline / start fast.
# YOLO weights + face ONNX go to /app/weights; EasyOCR models to /root/.EasyOCR.
RUN mkdir -p /app/weights/face && cd /app/weights \
    && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" \
    && python -c "import easyocr; easyocr.Reader(['en'], gpu=False)" \
    && curl -sSL -o /app/weights/face/yunet.onnx \
       "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    && curl -sSL -o /app/weights/face/sface.onnx \
       "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

COPY backend/ /app/backend/
COPY --from=frontend /frontend/dist /app/frontend/dist

ENV YOLO_MODEL=/app/weights/yolo11n.pt \
    YUNET_MODEL=/app/weights/face/yunet.onnx \
    SFACE_MODEL=/app/weights/face/sface.onnx \
    DATA_DIR=/app/data \
    PYTHONUNBUFFERED=1

EXPOSE 8000
# Single uvicorn worker: the detection manager owns child processes and shared
# state, which must live in one parent process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
