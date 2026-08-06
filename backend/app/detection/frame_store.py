"""Shared, cross-process state for running sources.

Each detection worker runs in its own process. The FastAPI process needs to
read the latest annotated JPEG frame (for MJPEG streaming), the live counters
and the runtime status of each source. We share these through a
``multiprocessing.Manager`` so both sides see the same data.

Only picklable, self-contained values are stored (bytes, plain dicts, tuples)
and whole values are reassigned on update to stay safe with Manager proxies.
"""
from __future__ import annotations

from multiprocessing.managers import SyncManager


class SharedState:
    def __init__(self, manager: SyncManager):
        # source_id -> latest annotated JPEG (bytes)
        self.frames = manager.dict()
        # source_id -> {class_name: {"in": n, "out": n}}
        self.live_counts = manager.dict()
        # source_id -> (status, message)
        self.status = manager.dict()

    # --- frames ---
    def set_frame(self, source_id: int, jpeg: bytes) -> None:
        self.frames[source_id] = jpeg

    def get_frame(self, source_id: int) -> bytes | None:
        return self.frames.get(source_id)

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

    def clear(self, source_id: int) -> None:
        for d in (self.frames, self.live_counts, self.status):
            if source_id in d:
                del d[source_id]
