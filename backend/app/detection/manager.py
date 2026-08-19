"""Supervises one detection worker process per running source."""
from __future__ import annotations

import multiprocessing as mp
import threading

from .frame_store import SharedState
from .worker import run_worker

# Set when the app is shutting down. MJPEG responses are endless generators;
# without this they keep yielding and uvicorn waits forever for the "in-flight"
# response to finish, so Ctrl-C appears to hang.
SHUTTING_DOWN = threading.Event()


class DetectionManager:
    def __init__(self):
        # "spawn" is required on Windows and is the safest cross-platform
        # choice given we load CUDA/torch inside workers.
        self._ctx = mp.get_context("spawn")
        self._manager = self._ctx.Manager()
        self.shared = SharedState(self._manager)
        # source_id -> (Process, stop Event)
        self._procs: dict[int, tuple] = {}

    def is_running(self, source_id: int) -> bool:
        entry = self._procs.get(source_id)
        return bool(entry and entry[0].is_alive())

    def running_ids(self) -> list[int]:
        return [sid for sid, (proc, _e) in self._procs.items() if proc.is_alive()]

    def start(self, source_cfg: dict) -> None:
        source_id = source_cfg["id"]
        if self.is_running(source_id):
            return
        # Clean up a dead entry if present.
        self._reap(source_id)

        stop_event = self._ctx.Event()
        proc = self._ctx.Process(
            target=run_worker,
            args=(source_cfg, self.shared, stop_event),
            name=f"worker-{source_id}",
            daemon=True,
        )
        proc.start()
        self._procs[source_id] = (proc, stop_event)

    def stop(self, source_id: int, timeout: float = 8.0) -> None:
        entry = self._procs.get(source_id)
        if not entry:
            return
        proc, stop_event = entry
        stop_event.set()
        proc.join(timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(3.0)
        self._procs.pop(source_id, None)
        self.shared.clear(source_id)

    def _reap(self, source_id: int) -> None:
        entry = self._procs.get(source_id)
        if entry and not entry[0].is_alive():
            self._procs.pop(source_id, None)

    def get_frame(self, source_id: int) -> bytes | None:
        return self.shared.get_frame(source_id)

    def get_counts(self, source_id: int) -> dict:
        return self.shared.get_counts(source_id)

    def shutdown(self) -> None:
        SHUTTING_DOWN.set()
        for source_id in list(self._procs.keys()):
            self.stop(source_id, timeout=4.0)
        try:
            self._manager.shutdown()
        except Exception:
            pass


# Global singleton, created at app startup.
manager: DetectionManager | None = None


def get_manager() -> DetectionManager:
    global manager
    if manager is None:
        raise RuntimeError("DetectionManager not initialized")
    return manager


def init_manager() -> DetectionManager:
    global manager
    if manager is None:
        manager = DetectionManager()
    return manager
