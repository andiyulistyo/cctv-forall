"""Collapse FFmpeg's per-frame decoder complaints into one summary line.

A camera that sends frames faster than this machine can take them ends up
dropping some of them on the wire, and the decoder then says so once per
missing reference picture:

    [hevc @ 0000023dae41de40] Could not find ref with POC 10
    [hevc @ 0000023dae428d80] Could not find ref with POC 12

Dozens of those a minute bury every other line in the console while saying the
same thing over and over. They cannot be filtered with `logging`: they come
from FFmpeg's C code, which writes straight to file descriptor 2 without ever
passing through Python. So the filtering happens there -- fd 2 is redirected
into a pipe, a thread reads it back, FFmpeg's own lines are counted instead of
printed, and everything else is passed through untouched.

The count is not thrown away: one line per interval reports how many messages
arrived and what the last one said, which is what an operator actually needs
(that the feed is losing frames, and roughly how badly). ``_CaptureHealth`` in
worker.py explains *why* on the source's status.
"""
from __future__ import annotations

import atexit
import os
import re
import threading
import time

# Every FFmpeg log line names the component that wrote it and the address of
# its context: "[hevc @ 0000023dae41de40] Could not find ref with POC 10".
# Matching that prefix is what tells FFmpeg's output apart from ours, and it
# also carries the part worth keeping -- the message after it.
_FFMPEG_PREFIX = re.compile(rb"^\[[\w .\-/]+ @ 0?x?[0-9a-fA-F]{6,}\]\s*")

_installed = False


def summarize_ffmpeg_noise(label: str = "", interval: float = 30.0) -> bool:
    """Send this process's fd 2 through the filter. True if it is in place.

    Only the first call in a process does anything; ``interval`` <= 0 leaves
    stderr alone, which is how an operator gets the raw lines back.
    """
    global _installed
    if _installed or interval <= 0:
        return False
    try:
        saved_fd = os.dup(2)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)
        os.close(write_fd)
        source = os.fdopen(read_fd, "rb")
    except OSError:
        # A process without a usable stderr (pythonw, a closed fd) is not worth
        # failing a worker over -- it has no console to flood either.
        return False
    _installed = True
    _Filter(label, interval, source, saved_fd).start()
    return True


class _Filter:
    """Reads the redirected stderr and decides what reaches the console."""

    def __init__(self, label: str, interval: float, source, out_fd: int):
        self._label = f"{label}: " if label else ""
        self._interval = interval
        self._source = source
        self._out_fd = out_fd
        self._lock = threading.Lock()
        self._count = 0
        self._latest = b""
        self._since = time.monotonic()

    def start(self) -> None:
        for target, name in ((self._pump, "ffmpeg-log"), (self._tick, "ffmpeg-log-tick")):
            threading.Thread(target=target, name=name, daemon=True).start()
        # Daemon threads are killed at shutdown, so the last few messages would
        # otherwise be counted and never reported.
        atexit.register(self._flush)

    def _write(self, data: bytes) -> None:
        try:
            os.write(self._out_fd, data)
        except OSError:
            pass

    def _pump(self) -> None:
        for line in self._source:
            if _FFMPEG_PREFIX.match(line):
                with self._lock:
                    self._count += 1
                    self._latest = _FFMPEG_PREFIX.sub(b"", line).strip()
                continue
            # Somebody else's output. Report what has piled up first so the
            # summary still lands next to the lines it belongs between.
            self._flush()
            self._write(line)

    def _tick(self) -> None:
        while True:
            time.sleep(self._interval)
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            count, latest, since = self._count, self._latest, self._since
            self._count, self._latest, self._since = 0, b"", time.monotonic()
        if not count:
            return
        seconds = max(1, round(time.monotonic() - since))
        message = latest.decode("utf-8", "replace")
        self._write(
            f"[ffmpeg] {self._label}{count} decoder message(s) in {seconds}s "
            f"(last: {message})\n".encode()
        )
