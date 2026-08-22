"""Tests for the dropped-frame diagnostics.

Two halves of the same problem: noticing that a camera's frames are going
missing (``_CaptureHealth``) and keeping FFmpeg's one-line-per-missing-picture
complaints from burying the log (``ffmpeg_log``).

    cd backend && PYTHONPATH=. python tests/test_capture_health.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.detection.ffmpeg_log import _FFMPEG_PREFIX  # noqa: E402
from app.detection.worker import _CAPTURE_WINDOWS, _CaptureHealth  # noqa: E402


class _FakeReader:
    """Just the one attribute _CaptureHealth reads off a SourceReader."""

    def __init__(self, fps: float):
        self.fps = fps


def _feed(health: _CaptureHealth, fps: float, windows: int, shape=(2160, 3840, 3)):
    message = None
    for _ in range(windows):
        message = health.update(fps, shape)
    return message


def test_healthy_stream_says_nothing():
    health = _CaptureHealth(_FakeReader(25.0))
    # A camera never lands exactly on its advertised rate; small change.
    assert _feed(health, 24.6, 10) is None


def test_sustained_shortfall_is_reported_with_what_to_do():
    health = _CaptureHealth(_FakeReader(25.0))
    # One slow window is not a verdict -- traffic bursts, keyframes, a GPU trim.
    assert _feed(health, 20.0, _CAPTURE_WINDOWS - 1) is None
    message = health.update(20.0, (2160, 3840, 3))
    assert message is not None
    assert "20 of 25 fps" in message and "3840x2160" in message
    # The operator needs the remedy, not just the symptom.
    assert "sub-stream" in message
    print(f"OK sustained_shortfall_is_reported: {message}")


def test_recovery_clears_the_warning():
    health = _CaptureHealth(_FakeReader(25.0))
    _feed(health, 20.0, _CAPTURE_WINDOWS)
    assert health.message is not None
    # Back at the stream's rate for a full averaging window.
    assert _feed(health, 25.0, _CAPTURE_WINDOWS) is None
    print("OK recovery_clears_the_warning")


def test_one_bad_window_among_good_ones_is_ignored():
    """The rate a busy worker reports is jumpy; the median is what counts."""
    health = _CaptureHealth(_FakeReader(25.0))
    _feed(health, 25.0, _CAPTURE_WINDOWS)
    assert health.update(3.0, (720, 1280, 3)) is None  # one stalled window
    assert _feed(health, 25.0, 2) is None
    print("OK one_bad_window_among_good_ones_is_ignored")


def test_flapping_rate_does_not_flip_the_status():
    """A rate sitting between the two thresholds keeps whatever it had.

    Every change writes the source row, so a stream hovering at 23 of 25 fps
    must not churn the database once per stat window.
    """
    health = _CaptureHealth(_FakeReader(25.0))
    assert _feed(health, 24.0, 20) is None  # 0.96 -- short, but not lagging
    _feed(health, 20.0, _CAPTURE_WINDOWS)
    warned = health.message
    assert warned is not None
    # Same wording, so the source row is written once and not once a window.
    assert _feed(health, 24.0, 20) == warned
    print("OK flapping_rate_does_not_flip_the_status")


def test_unknown_stream_rate_is_not_judged():
    # Plenty of streams report 0 or nonsense; there is nothing to compare to.
    health = _CaptureHealth(_FakeReader(0.0))
    assert _feed(health, 2.0, 10) is None
    print("OK unknown_stream_rate_is_not_judged")


def test_only_ffmpegs_own_lines_match_the_filter():
    noise = [
        b"[hevc @ 0000023dae41de40] Could not find ref with POC 10\n",
        b"[h264 @ 000001c47f9fe1c0] error while decoding MB 42 12\n",
        b"[rtsp @ 0000023b3398e940] max delay reached. need to consume packet\n",
        b"[swscaler @ 000001c41e2da840] deprecated pixel format used\n",
    ]
    ours = [
        b"[SourceReader] hwaccel 'auto' failed for rtsp source\n",
        b"[preview] frame dropped: ValueError: bad shape\n",
        b"INFO:     127.0.0.1:54767 - \"GET /sources/6 HTTP/1.1\" 200 OK\n",
        b"Traceback (most recent call last):\n",
    ]
    assert all(_FFMPEG_PREFIX.match(line) for line in noise)
    assert not any(_FFMPEG_PREFIX.match(line) for line in ours)
    print("OK only_ffmpegs_own_lines_match_the_filter")


def test_filter_collapses_the_flood_and_passes_the_rest_through():
    """End to end, in a child process: its fd 2 is the thing being replaced."""
    script = """
import os, sys, time
from app.detection.ffmpeg_log import summarize_ffmpeg_noise
assert summarize_ffmpeg_noise("source 6", interval=0.2)
for i in range(50):
    os.write(2, b"[hevc @ 0000023dae41de40] Could not find ref with POC 44\\n")
time.sleep(0.5)
sys.stderr.write("worker still talking\\n")
sys.stderr.flush()
time.sleep(0.2)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(BACKEND),
        capture_output=True,
        timeout=60,
    )
    lines = [ln for ln in proc.stderr.decode().splitlines() if ln.strip()]
    assert not any("Could not find ref" in ln and ln.startswith("[hevc") for ln in lines), lines
    summaries = [ln for ln in lines if ln.startswith("[ffmpeg] source 6:")]
    assert len(summaries) == 1, lines
    assert "50 decoder message(s)" in summaries[0], summaries
    assert "worker still talking" in lines, lines
    print(f"OK filter_collapses_the_flood: {summaries[0]}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} capture-health tests passed.")
