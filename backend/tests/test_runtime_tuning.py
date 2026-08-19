"""Tests for the performance-tuning helpers (no torch/OpenCV required).

    cd backend && PYTHONPATH=. python tests/test_runtime_tuning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detection.detector import Detection, is_non_torch_model  # noqa: E402
from app.detection.source_reader import (  # noqa: E402
    CHUNKED_SOURCE_TYPES,
    REALTIME_SOURCE_TYPES,
    _capture_options,
    _yt_format,
)
from app.runtime import threads_per_worker  # noqa: E402


def test_explicit_thread_count_wins():
    assert threads_per_worker(6, expected_streams=4, gpu=True) == 6
    assert threads_per_worker(6, expected_streams=1, gpu=False) == 6
    print("OK explicit_thread_count_wins")


def test_threads_split_across_expected_streams():
    # 8 performance cores / 4 streams = 2 each; never below 1.
    got = threads_per_worker(0, expected_streams=4, gpu=False)
    assert got >= 1, got
    assert threads_per_worker(0, expected_streams=1000, gpu=False) == 1
    print("OK threads_split_across_expected_streams:", got)


def test_gpu_workers_get_a_small_cpu_budget():
    # With inference on the GPU the CPU only pre/post-processes.
    assert threads_per_worker(0, expected_streams=1, gpu=True) <= 2
    print("OK gpu_workers_get_a_small_cpu_budget")


def test_capture_options():
    rtsp = _capture_options("rtsp", "videotoolbox", True)
    assert rtsp.startswith("hwaccel;videotoolbox|rtsp_transport;tcp"), rtsp
    # Real-time protocols also get the low-latency flags...
    assert "fflags;nobuffer" in rtsp and "flags;low_delay" in rtsp
    # A silently dropped camera must time out instead of blocking forever.
    assert "timeout;" in rtsp
    # ...but HLS must keep its buffer, or every segment boundary stutters.
    hls = _capture_options("hls", "videotoolbox", True)
    assert hls == "hwaccel;videotoolbox", hls
    # TCP transport is an RTSP-only option.
    assert _capture_options("http", "", True) == ""
    print("OK capture_options")


def test_source_types_are_disjoint():
    # A source is either paced by the sender or downloaded in chunks, never
    # both — the worker picks its buffering strategy from these.
    assert not set(REALTIME_SOURCE_TYPES) & set(CHUNKED_SOURCE_TYPES)
    assert "rtsp" in REALTIME_SOURCE_TYPES and "youtube" in CHUNKED_SOURCE_TYPES
    print("OK source_types_are_disjoint")


def test_youtube_format_caps_resolution():
    fmt = _yt_format(720)
    assert "[height<=720]" in fmt
    # A capped HLS stream must be preferred over an uncapped fallback.
    assert fmt.index("[height<=720]") < fmt.index("best[protocol=m3u8_native]/bestvideo")
    assert "[height<=480]" in _yt_format(480)
    print("OK youtube_format_caps_resolution")


def test_non_torch_model_detection():
    assert is_non_torch_model("data/weights/yolo11n.mlpackage")
    assert is_non_torch_model("/abs/path/model.onnx")
    assert not is_non_torch_model("yolo11n.pt")
    print("OK non_torch_model_detection")


def test_detection_scaled_back_to_full_frame():
    # Boxes found on a 960px-wide frame must map onto the 1920px original.
    det = Detection(1, "car", 0.9, 10.0, 20.0, 50.0, 60.0)
    big = det.scaled(2.0)
    assert (big.x1, big.y1, big.x2, big.y2) == (20.0, 40.0, 100.0, 120.0)
    assert big.track_id == det.track_id and big.class_name == det.class_name
    assert det.scaled(1.0) is det  # no copy when nothing changes
    print("OK detection_scaled_back_to_full_frame")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} runtime-tuning tests passed.")
