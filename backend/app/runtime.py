"""Hardware detection and per-process runtime tuning.

Kept free of heavy imports (no cv2 / torch at module level) so it can be used
very early in a worker process, before those libraries are configured.

The main consumer is the detection worker: every source runs in its own
process, so without an explicit thread budget OpenCV and torch each spin up
one thread per core *per worker*. On a machine like the Mac mini M4 Pro
(8 performance + 4 efficiency cores) four streams would fight over ~48 threads
and lose more time to context switching than to actual inference.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from functools import lru_cache


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_apple_silicon() -> bool:
    return is_macos() and platform.machine() == "arm64"


@lru_cache(maxsize=None)
def _sysctl_int(key: str) -> int | None:
    if not is_macos():
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=2
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip())
    except Exception:
        pass
    return None


@lru_cache(maxsize=None)
def performance_cores() -> int:
    """Number of performance ("P") cores, falling back to total cores.

    Apple Silicon exposes core clusters via ``hw.perflevel0`` (performance) and
    ``hw.perflevel1`` (efficiency). Scheduling detection work as if the E-cores
    were full-speed cores just adds contention, so we budget on P-cores only.
    """
    cores = _sysctl_int("hw.perflevel0.logicalcpu")
    if cores:
        return cores
    return os.cpu_count() or 4


@lru_cache(maxsize=None)
def chip_name() -> str:
    if is_macos():
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def has_mps() -> bool:
    """True when this machine can run torch on the Apple GPU (Metal)."""
    if not is_apple_silicon():
        return False
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def threads_per_worker(configured: int, expected_streams: int, gpu: bool) -> int:
    """Resolve the CPU thread budget for a single detection worker.

    ``configured`` > 0 wins. Otherwise the P-cores are split across the number
    of streams the operator expects to run, leaving headroom for the API
    process and for video decoding. When inference runs on the GPU (MPS/CUDA)
    the CPU only does pre/post-processing, so a small budget is plenty.
    """
    if configured > 0:
        return configured
    cores = performance_cores()
    streams = max(1, expected_streams)
    budget = max(1, cores // streams)
    if gpu:
        # GPU does the matmuls; more CPU threads only add contention.
        budget = min(budget, 2)
    return max(1, min(budget, cores))


def apply_worker_threads(n_threads: int) -> None:
    """Pin OpenCV/torch/BLAS thread pools for the current process."""
    n = max(1, int(n_threads))
    # Affects BLAS backends that read these at first use.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)
    try:
        import cv2

        # OpenCV's own pool: decoding, resize, JPEG encode. One thread per
        # worker keeps N workers from each grabbing every core.
        cv2.setNumThreads(n)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        # set_num_interop_threads raises if called after parallel work started.
        pass


def describe() -> dict:
    """Small summary used by the /api/health endpoint."""
    info = {
        "platform": sys.platform,
        "machine": platform.machine(),
        "cpu": chip_name(),
        "cpu_cores": os.cpu_count(),
        "performance_cores": performance_cores(),
        "apple_silicon": is_apple_silicon(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:
        info["torch"] = None
    return info
