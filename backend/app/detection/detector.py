"""Ultralytics YOLO wrapper with built-in ByteTrack tracking.

Automatically selects CUDA when available, otherwise CPU. Detection is
restricted to the classes the user enabled for a source.
"""
from __future__ import annotations

from dataclasses import dataclass

# COCO class id -> our friendly class name (only the ones we care about).
COCO_ID_TO_NAME = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
NAME_TO_COCO_ID = {v: k for k, v in COCO_ID_TO_NAME.items()}


def pick_device(preferred: str = "") -> str:
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


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


class Detector:
    def __init__(self, model_path: str, device: str = "", conf: float = 0.35, imgsz: int = 640):
        from ultralytics import YOLO

        self.device = pick_device(device)
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model_path)

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
