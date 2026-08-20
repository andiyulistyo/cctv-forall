"""Tests for the performance-tuning helpers.

Only the OpenCV import is required (for the hardware-acceleration constants);
no torch, no OpenVINO.

    cd backend && PYTHONPATH=. python tests/test_runtime_tuning.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from app import runtime  # noqa: E402
from app.detection import detector as detector_mod  # noqa: E402
from app.detection.detector import (  # noqa: E402
    Detection,
    is_non_torch_model,
    is_openvino_model,
    plan_inference,
)
from app.detection.source_reader import (  # noqa: E402
    CHUNKED_SOURCE_TYPES,
    REALTIME_SOURCE_TYPES,
    _ACCELERATION_TYPES,
    _capture_options,
    _HWACCEL_MIN_HEALTHY_FRAMES,
    _hw_acceleration,
    _yt_format,
    SourceReader,
)
from app.config import settings  # noqa: E402
from app.detection.worker import _ocr_device  # noqa: E402
from app.runtime import cpu_affinity_slice, physical_cores, threads_per_worker  # noqa: E402


class _intel_gpu:
    """Pretend this machine does / does not expose an OpenVINO GPU."""

    def __init__(self, present: bool):
        self.present = present

    def __enter__(self):
        self._saved = runtime.has_intel_gpu
        runtime.has_intel_gpu = lambda: self.present
        detector_mod.runtime.has_intel_gpu = runtime.has_intel_gpu
        return self

    def __exit__(self, *exc):
        runtime.has_intel_gpu = self._saved
        detector_mod.runtime.has_intel_gpu = self._saved


def test_explicit_thread_count_wins():
    assert threads_per_worker(6, expected_streams=4, gpu=True) == 6
    assert threads_per_worker(6, expected_streams=1, gpu=False) == 6
    print("OK explicit_thread_count_wins")


def test_threads_split_across_expected_streams():
    # 8 physical cores / 4 streams = 2 each; never below 1.
    got = threads_per_worker(0, expected_streams=4, gpu=False)
    assert got >= 1, got
    assert threads_per_worker(0, expected_streams=1000, gpu=False) == 1
    print("OK threads_split_across_expected_streams:", got)


def test_gpu_workers_get_a_small_cpu_budget():
    # With inference on the GPU the CPU only pre/post-processes.
    assert threads_per_worker(0, expected_streams=1, gpu=True) <= 2
    print("OK gpu_workers_get_a_small_cpu_budget")


def test_physical_cores_ignores_smt_siblings():
    logical = os.cpu_count() or 1
    cores = physical_cores()
    assert 1 <= cores <= logical, (cores, logical)
    print(f"OK physical_cores_ignores_smt_siblings: {cores} of {logical} logical")


def test_affinity_slices_do_not_overlap():
    # Two threads per worker on an 8-core / 16-thread machine => 4 disjoint
    # slices, each holding both SMT siblings of its cores.
    slices = [cpu_affinity_slice(slot, 2) for slot in range(4)]
    if slices[0] is None:
        print("OK affinity_slices_do_not_overlap: skipped (too few cores)")
        return
    seen: set[int] = set()
    for s in slices:
        assert s, s
        assert not (seen & set(s)), (seen, s)
        seen |= set(s)
    # A worker that would own every core is left unpinned instead.
    assert cpu_affinity_slice(0, physical_cores()) is None
    print("OK affinity_slices_do_not_overlap:", slices)


def test_capture_options():
    # "videotoolbox" has no OpenCV acceleration constant, so it keeps going
    # through the FFmpeg option string.
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


def test_hwaccel_modes_use_the_opencv_api():
    # These are real OpenCV acceleration types and must NOT be smuggled into
    # the FFmpeg option string, where "hwaccel" is not a valid AVFormat option.
    assert _hw_acceleration("auto") == cv2.VIDEO_ACCELERATION_ANY
    assert _hw_acceleration("d3d11va") == cv2.VIDEO_ACCELERATION_D3D11
    assert _hw_acceleration("qsv") == cv2.VIDEO_ACCELERATION_MFX
    assert _hw_acceleration("vaapi") == cv2.VIDEO_ACCELERATION_VAAPI
    assert _hw_acceleration("off") == cv2.VIDEO_ACCELERATION_NONE
    assert _hw_acceleration("") == cv2.VIDEO_ACCELERATION_NONE
    assert "hwaccel" not in _capture_options("rtsp", "auto", False)
    assert "hwaccel" not in _capture_options("rtsp", "d3d11va", False)
    # Unknown to OpenCV -> handed to FFmpeg, as before.
    assert _hw_acceleration("cuda") is None
    assert "hwaccel;cuda" in _capture_options("rtsp", "cuda", False)
    print("OK hwaccel_modes_use_the_opencv_api")


def test_hwaccel_survives_an_opencv_without_every_constant():
    # Builds differ in which acceleration types they expose (DRM only landed in
    # OpenCV 4.12). A missing one must degrade to the FFmpeg option string, not
    # raise AttributeError inside a running worker.
    for mode in list(_ACCELERATION_TYPES) + ["off", "", "cuda", "videotoolbox", "bogus"]:
        _hw_acceleration(mode)          # must not raise
        _capture_options("rtsp", mode, True)
    missing = [m for m, name in _ACCELERATION_TYPES.items()
               if not hasattr(cv2, f"VIDEO_ACCELERATION_{name}")]
    for mode in missing:
        assert _hw_acceleration(mode) is None
        assert f"hwaccel;{mode}" in _capture_options("rtsp", mode, False)
    # Turning it off must never leak a bogus "hwaccel;off" into FFmpeg, even if
    # a build somehow lacks VIDEO_ACCELERATION_NONE.
    for mode in ("off", "", "none", "software"):
        assert "hwaccel" not in _capture_options("rtsp", mode, False)
    print(f"OK hwaccel_survives_an_opencv_without_every_constant "
          f"(cv2 {cv2.__version__}, missing: {missing or 'none'})")


def _reader_after(sessions):
    """Replay a list of per-session frame counts through the hwaccel guard.

    Each entry is how many frames one capture delivered before its read failed.
    Returns the reader so the caller can inspect whether hardware decoding
    survived.
    """
    reader = SourceReader("rtsp", "rtsp://cam/stream", hwaccel="auto")
    for frames in sessions:
        reader._frames_since_open = frames
        reader._note_failed_session()
    return reader


def test_hwaccel_falls_back_when_the_decoder_keeps_giving_up():
    # D3D11VA on an Intel iGPU opens an HEVC stream happily and only fails on
    # the first picture needing a reference frame, so the failure shows up as
    # a capture that dies almost immediately. Two of those in a row and we
    # stop paying for the hardware path.
    assert not _reader_after([0, 0])._hwaccel_ok
    assert not _reader_after([1, 2])._hwaccel_ok
    # One short session on its own is not enough to conclude anything.
    assert _reader_after([0])._hwaccel_ok
    print("OK hwaccel_falls_back_when_the_decoder_keeps_giving_up")


def test_a_flapping_camera_is_not_mistaken_for_a_broken_decoder():
    # A stream that plays and then drops is a network problem; hardware
    # decoding must survive it however often it happens.
    healthy = _HWACCEL_MIN_HEALTHY_FRAMES
    assert _reader_after([healthy] * 5)._hwaccel_ok
    # ...and a good session clears the short ones that came before it, so
    # unrelated drops never add up to a false diagnosis.
    assert _reader_after([0, healthy, 0])._hwaccel_ok
    print("OK a_flapping_camera_is_not_mistaken_for_a_broken_decoder")


def test_software_decoding_needs_no_fallback():
    reader = SourceReader("rtsp", "rtsp://cam/stream", hwaccel="off")
    assert not reader._hwaccel_ok
    # Nothing to fall back to, so a failed session must not ask the read loop
    # to skip its reconnect delay and spin.
    reader._frames_since_open = 0
    assert reader._note_failed_session() is False
    print("OK software_decoding_needs_no_fallback")


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


def test_openvino_model_detection():
    # The ultralytics OpenVINO export is a *directory*, so a suffix check alone
    # would miss it and the model would be treated as torch weights.
    assert is_openvino_model("data/weights/yolo11n_openvino_model")
    assert is_openvino_model("data/weights/yolo11n_openvino_model/")
    assert is_openvino_model("/abs/path/yolo11s_int8_openvino_model")
    assert is_openvino_model("data/weights/model.xml")
    assert not is_openvino_model("yolo11n.pt")
    assert not is_openvino_model("data/weights/yolo11n.mlpackage")
    # ...and it counts as non-torch everywhere else too.
    assert is_non_torch_model("data/weights/yolo11n_openvino_model")
    print("OK openvino_model_detection")


def test_openvino_device_follows_the_hardware():
    ov = "data/weights/yolo11n_openvino_model"
    with _intel_gpu(True):
        assert plan_inference(ov).device == "intel:gpu"
    with _intel_gpu(False):
        assert plan_inference(ov).device == "intel:cpu"
        # An explicit setting always wins over auto-detection.
        assert plan_inference(ov, "intel:gpu").device == "intel:gpu"
    print("OK openvino_device_follows_the_hardware")


def test_openvino_cpu_worker_keeps_the_full_thread_budget():
    ov = "data/weights/yolo11n_openvino_model"
    with _intel_gpu(False):
        cpu_plan = plan_inference(ov)
    with _intel_gpu(True):
        gpu_plan = plan_inference(ov)
    # The CPU plugin does the whole inference on the cores: capping it at the
    # 2-thread accelerator budget would be exactly backwards.
    assert cpu_plan.cpu_bound and cpu_plan.backend == "openvino"
    assert not gpu_plan.cpu_bound and gpu_plan.backend == "openvino"
    # Precision is baked into the IR, so half is never claimed.
    assert not plan_inference(ov, "intel:cpu", half=True).half
    # CoreML is an accelerator; plain torch on the CPU is not.
    assert not plan_inference("yolo11n.mlpackage").cpu_bound
    assert plan_inference("yolo11n.pt", "cpu").cpu_bound
    assert not plan_inference("yolo11n.pt", "cuda").cpu_bound
    # fp16 stays a GPU-only claim.
    assert plan_inference("yolo11n.pt", "cuda", half=True).half
    assert not plan_inference("yolo11n.pt", "cpu", half=True).half
    print("OK openvino_cpu_worker_keeps_the_full_thread_budget")


def test_ocr_device_never_inherits_a_non_torch_device():
    # Exercise the fallback, not whatever OCR_DEVICE the local .env happens to
    # set -- an explicit setting is meant to win and would mask the logic.
    saved, settings.ocr_device = settings.ocr_device, ""
    try:
        # EasyOCR is a torch model: "intel:gpu" / "mps" mean nothing to it.
        assert _ocr_device("intel:gpu") == "cpu"
        assert _ocr_device("intel:cpu") == "cpu"
        assert _ocr_device("mps") == "cpu"
        assert _ocr_device("cpu") == "cpu"
        # CUDA is the one device worth sharing.
        assert _ocr_device("cuda") == "cuda"
        # ...and an explicit setting still overrides everything.
        settings.ocr_device = "cpu"
        assert _ocr_device("cuda") == "cpu"
    finally:
        settings.ocr_device = saved
    print("OK ocr_device_never_inherits_a_non_torch_device")


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
