#!/usr/bin/env python
"""Check the ONNX Runtime face backend against the OpenCV reference.

``onnx_face.py`` re-implements YuNet's anchor decoding by hand, which is the
one genuinely risky part of moving face recognition off ``cv2.dnn``: a subtly
wrong decode does not crash, it just quietly detects the wrong things.

Two separate checks, because they answer different questions:

1. **Decode equivalence (strict, must pass).** Both implementations are given
   the *identical* input tensor, so the only thing under test is the anchor
   decoding and NMS. Boxes, landmarks and scores must agree to within a
   fraction of a pixel.

2. **End-to-end comparison (informational).** The two full pipelines are run
   on real images. Boxes and landmarks should agree closely; the *embeddings*
   will not. cv2.dnn's evaluation of SFace deviates from ONNX Runtime's, so
   even a byte-identical input blob yields ~0.93 cosine rather than 1.0 (ORT on
   CPU and on CUDA agree with each other exactly, so this is the runtime, not
   the device). That is why switching FACE_BACKEND requires re-enrolling every
   face -- see the README.

    cd backend && .venv/Scripts/python ../scripts/validate_onnx_face.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# Check 1 is a numerical-identity test: anything above float noise is a bug.
_MAX_DECODE_DELTA_PX = 0.5
_MAX_DECODE_SCORE_DELTA = 1e-3


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a[:4]
    bx1, by1, bw, bh = b[:4]
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / ((np.linalg.norm(u) + 1e-9) * (np.linalg.norm(v) + 1e-9)))


def ensure_test_images(face_dir: Path) -> list[Path]:
    """A single portrait plus a multi-face canvas built from it.

    The canvas exercises multiple scales and the NMS path, which a single
    centred portrait does not.
    """
    import urllib.request

    lena = face_dir / "_test_lena.jpg"
    if not lena.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            lena,
        )
    canvas_path = face_dir / "_test_faces.jpg"
    if not canvas_path.exists():
        im = cv2.imread(str(lena))
        # Locate the face rather than hard-coding a crop: a wrong guess yields
        # a canvas with no faces at all, which silently passes every check.
        det = cv2.FaceDetectorYN.create(
            str(face_dir / "yunet.onnx"), "", (im.shape[1], im.shape[0]), 0.7, 0.3, 5000
        )
        _, faces = det.detect(im)
        if faces is None or len(faces) == 0:
            raise SystemExit("could not locate a face in the reference portrait")
        x, y, w, h = faces[0][:4].astype(int)
        pad = int(0.25 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(im.shape[1], x + w + pad), min(im.shape[0], y + h + pad)
        face = im[y0:y1, x0:x1]

        canvas = np.full((720, 1280, 3), 90, np.uint8)
        for (px, py, sc) in [(60, 40, 1.0), (430, 60, 0.75), (800, 40, 0.55),
                             (170, 400, 0.5), (650, 380, 0.85)]:
            f = cv2.resize(face, None, fx=sc, fy=sc)
            fh, fw = f.shape[:2]
            if py + fh <= 720 and px + fw <= 1280:
                canvas[py:py + fh, px:px + fw] = f
        cv2.imwrite(str(canvas_path), canvas)
    return [canvas_path, lena]


def check_decode(new, yunet_path: str, images: list[Path]) -> int:
    """Check 1: identical input tensor, so only the decode differs."""
    ref = cv2.FaceDetectorYN.create(
        yunet_path, "", (320, 320), new.score_threshold, new.nms_threshold, new.top_k,
    )

    failures = 0
    print("Check 1 - decode equivalence on an identical input tensor")
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        prepped, scale = new._preprocess(img)
        h, w = prepped.shape[:2]
        ref.setInputSize((w, h))
        _, ref_rows = ref.detect(prepped)
        ref_rows = [] if ref_rows is None else list(ref_rows)

        # Our rows are in original-frame coords; undo the scale to compare in
        # the same space cv2 reported.
        mine = [r.copy() for r in new.detect(img)]
        for q in mine:
            q[:14] *= scale

        if len(ref_rows) != len(mine):
            print(f"  !! {path.name}: cv2={len(ref_rows)} onnx={len(mine)} faces")
            failures += 1
            continue

        worst_box = worst_lm = worst_score = 0.0
        for a, b in zip(ref_rows, mine):
            worst_box = max(worst_box, float(np.max(np.abs(a[:4] - b[:4]))))
            worst_lm = max(worst_lm, float(np.max(np.abs(a[4:14] - b[4:14]))))
            worst_score = max(worst_score, abs(float(a[14]) - float(b[14])))
        ok = (worst_box <= _MAX_DECODE_DELTA_PX
              and worst_lm <= _MAX_DECODE_DELTA_PX
              and worst_score <= _MAX_DECODE_SCORE_DELTA)
        print(f"  {'ok ' if ok else '!! '}{path.name:20} {len(mine)} faces  "
              f"input {w}x{h}  box delta {worst_box:.3f}px  "
              f"landmark delta {worst_lm:.3f}px  score delta {worst_score:.2e}")
        if not ok:
            failures += 1
    return failures


def report_end_to_end(ref, new, images: list[Path]) -> None:
    """Check 2: full pipelines, differences expected and reported, not failed.

    Boxes and landmarks should agree closely. The embeddings will not: cv2.dnn
    evaluates SFace differently from ONNX Runtime, which is the reason
    switching FACE_BACKEND means re-enrolling.
    """
    print("\nCheck 2 - full pipeline: cv2.dnn vs ONNX Runtime, end to end")
    print("  (boxes should match; embedding cosine ~0.93 is expected, not a fault)")
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        ref_faces, new_faces = ref.detect(img), new.detect(img)
        print(f"  {path.name:20} cv2={len(ref_faces)} onnx={len(new_faces)} faces")
        unmatched = list(range(len(new_faces)))
        for rf in ref_faces:
            if not unmatched:
                break
            j = max(unmatched, key=lambda k: iou(rf, new_faces[k]))
            unmatched.remove(j)
            nf = new_faces[j]
            e1, e2 = ref.embed(img, rf), new.embed(img, nf)
            cos = cosine(e1, e2) if e1 is not None and e2 is not None else float("nan")
            lm = float(np.max(np.abs(np.asarray(rf[4:14]) - np.asarray(nf[4:14]))))
            print(f"      IoU={iou(rf, nf):.4f}  landmark delta={lm:5.2f}px  "
                  f"embedding cosine={cos:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="", help="extra image to test")
    args = ap.parse_args()

    from app.config import settings
    from app.detection.face import FaceRecognizer
    from app.detection.onnx_face import OnnxFaceBackend

    face_dir = Path(settings.yunet_model_path).parent
    images = ensure_test_images(face_dir)
    if args.image:
        images.insert(0, Path(args.image))

    # Force the OpenCV backend: FaceRecognizer now prefers ONNX, so without
    # this the "reference" would be the very thing under test and every
    # comparison would trivially report a perfect match.
    ref = FaceRecognizer(
        settings.yunet_model_path, settings.sface_model_path, backend="opencv"
    )
    if ref.backend != "opencv":
        raise SystemExit("could not construct the OpenCV reference backend")
    new = OnnxFaceBackend(settings.yunet_model_path, settings.sface_model_path)
    print(f"reference: cv2.FaceDetectorYN     new: onnxruntime [{new.provider}]\n")

    failures = check_decode(new, settings.yunet_model_path, images)
    report_end_to_end(ref, new, images)

    print()
    if failures:
        print(f"FAILED: {failures} decode mismatch(es).")
        raise SystemExit(1)
    print("Decode matches the OpenCV reference.")


if __name__ == "__main__":
    main()
