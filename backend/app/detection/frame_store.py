"""Shared, cross-process state for running sources.

Each detection worker runs in its own process. The FastAPI process needs to
read the latest annotated JPEG frame (for MJPEG streaming), the live counters
and the runtime status of each source. We share these through a
``multiprocessing.Manager`` so both sides see the same data.

Only picklable, self-contained values are stored (bytes, plain dicts, tuples)
and whole values are reassigned on update to stay safe with Manager proxies.

Every proxy access is a pickle round-trip over a socket to the manager
process, and a JPEG is tens of kilobytes. Frames therefore carry a sequence
number in a separate (tiny) dict, so a streaming client can poll the counter
cheaply and only pull the image when it has actually changed.
"""
from __future__ import annotations

from multiprocessing.managers import SyncManager


class SharedState:
    def __init__(self, manager: SyncManager):
        # source_id -> latest annotated JPEG (bytes)
        self.frames = manager.dict()
        # source_id -> counter incremented on every published frame
        self.frame_seq = manager.dict()
        # source_id -> {class_name: {"in": n, "out": n}}
        self.live_counts = manager.dict()
        # source_id -> (status, message)
        self.status = manager.dict()
        # source_id -> {"capture_fps": f, "detect_fps": f, "publish_fps": f}
        self.stats = manager.dict()

    # --- frames ---
    def set_frame(self, source_id: int, jpeg: bytes, seq: int) -> None:
        self.frames[source_id] = jpeg
        self.frame_seq[source_id] = seq

    def get_frame(self, source_id: int) -> bytes | None:
        return self.frames.get(source_id)

    def get_frame_seq(self, source_id: int) -> int:
        """Cheap "has the frame changed?" check for streaming clients."""
        return self.frame_seq.get(source_id, 0)

    # --- counts ---
    def set_counts(self, source_id: int, counts: dict) -> None:
        self.live_counts[source_id] = counts

    def get_counts(self, source_id: int) -> dict:
        return self.live_counts.get(source_id, {})

    # --- status ---
    def set_status(self, source_id: int, status: str, message: str | None = None) -> None:
        self.status[source_id] = (status, message)

    def get_status(self, source_id: int):
        return self.status.get(source_id)

    # --- throughput stats (diagnostics) ---
    def set_stats(self, source_id: int, stats: dict) -> None:
        self.stats[source_id] = stats

    def get_stats(self, source_id: int) -> dict | None:
        return self.stats.get(source_id)

    def clear(self, source_id: int) -> None:
        for d in (self.frames, self.frame_seq, self.live_counts, self.status, self.stats):
            if source_id in d:
                del d[source_id]
