"""Hardware detection and per-process runtime tuning.

Kept free of heavy imports (no cv2 / torch / openvino at module level) so it
can be used very early in a worker process, before those libraries are
configured.

The main consumer is the detection worker: every source runs in its own
process, so without an explicit thread budget OpenCV and torch each spin up
one thread per core *per worker*. On a machine like the Mac mini M4 Pro
(8 performance + 4 efficiency cores) four streams would fight over ~48 threads
and lose more time to context switching than to actual inference. The same
applies to a Ryzen 7840U (8 cores / 16 threads) on Windows.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from functools import lru_cache


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_windows() -> bool:
    return sys.platform == "win32"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


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
def _psutil_physical_cores() -> int | None:
    """Physical (non-SMT) core count, or None when psutil is unavailable.

    psutil comes along with ultralytics, so this costs no extra dependency.
    It matters because ``os.cpu_count()`` reports *logical* CPUs: budgeting 16
    threads on an 8-core Ryzen just makes the SMT siblings fight each other.
    """
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
        return int(cores) if cores else None
    except Exception:
        return None


@lru_cache(maxsize=None)
def physical_cores() -> int:
    """Cores worth scheduling detection work on.

    Apple Silicon exposes core clusters via ``hw.perflevel0`` (performance) and
    ``hw.perflevel1`` (efficiency). Scheduling detection work as if the E-cores
    were full-speed cores just adds contention, so we budget on P-cores only.
    Elsewhere we budget on physical cores and ignore SMT siblings.
    """
    cores = _sysctl_int("hw.perflevel0.logicalcpu")
    if cores:
        return cores
    cores = _psutil_physical_cores()
    if cores:
        return cores
    return os.cpu_count() or 4


# Historic name, still used by scripts/benchmark.py and the tests.
performance_cores = physical_cores


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
    if is_windows():
        # platform.processor() on Windows returns the family/model string
        # ("AMD64 Family 25 Model 116 ..."), not the marketing name we want in
        # the health payload and in the setup script.
        try:
            out = subprocess.run(
                [
                    "powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def cpu_vendor() -> str:
    """One of ``apple`` / ``amd`` / ``intel`` / ``unknown``.

    Used to pick a starting profile in the setup script and to report what the
    app thinks it is running on.
    """
    if is_apple_silicon():
        return "apple"
    name = f"{chip_name()} {platform.processor()}".lower()
    if "amd" in name or "ryzen" in name:
        return "amd"
    if "intel" in name or "core(tm)" in name:
        return "intel"
    return "unknown"


def has_mps() -> bool:
    """True when this machine can run torch on the Apple GPU (Metal)."""
    if not is_apple_silicon():
        return False
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def has_cuda() -> bool:
    """True when torch can run on an NVIDIA GPU.

    Deliberately not cached: a CPU-only torch wheel is the usual reason this is
    False, and re-installing the CUDA wheel is exactly what the operator does
    next -- caching False across a long-lived process would hide the fix.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cuda_devices() -> tuple[dict, ...]:
    """One entry per visible NVIDIA GPU, empty when there is none.

    ``capability`` is the compute capability ("12.0" on Blackwell). It is worth
    reporting because it is what decides whether the installed wheel can run
    here at all: an RTX 50-series card is sm_120 and needs a CUDA 12.8+ build,
    and an older wheel fails at the first kernel launch rather than at import.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return ()
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            out.append({
                "index": i,
                "name": props.name,
                "total_memory_mb": round(props.total_memory / (1024 * 1024)),
                "capability": f"{props.major}.{props.minor}",
            })
        return tuple(out)
    except Exception:
        return ()


def gpu_vendor() -> str:
    """Accelerator vendor to build a profile around: ``nvidia`` / ``intel`` /
    ``apple`` / ``none``.

    Checked *before* :func:`cpu_vendor` by the setup scripts. The two disagree
    often -- an NVIDIA laptop usually has an AMD or Intel CPU -- and picking the
    profile from the CPU there would set up the OpenVINO CPU plugin and leave
    the discrete GPU idle.
    """
    if has_cuda():
        return "nvidia"
    if is_apple_silicon():
        return "apple"
    if has_intel_gpu():
        return "intel"
    return "none"


@lru_cache(maxsize=None)
def openvino_devices() -> tuple[str, ...]:
    """OpenVINO devices present on this machine, e.g. ``("CPU", "GPU")``.

    Empty when the openvino package is not installed (it is optional -- see
    backend/requirements-openvino.txt).
    """
    try:
        import openvino as ov

        return tuple(ov.Core().available_devices)
    except Exception:
        return ()


def has_intel_gpu() -> bool:
    """True when OpenVINO can target an Intel iGPU/dGPU.

    Intel-only by construction: the OpenVINO GPU plugin does not support AMD
    Radeon, so a Ryzen machine reports False and runs on the CPU plugin.
    """
    return any(d == "GPU" or d.startswith("GPU.") for d in openvino_devices())


@lru_cache(maxsize=None)
def openvino_version() -> str | None:
    try:
        import openvino as ov

        return str(ov.__version__)
    except Exception:
        return None


def threads_per_worker(
    configured: int,
    expected_streams: int,
    gpu: bool,
    *,
    gpu_thread_cap: int = 2,
) -> int:
    """Resolve the CPU thread budget for a single detection worker.

    ``configured`` > 0 wins. Otherwise the cores are split across the number of
    streams the operator expects to run, leaving headroom for the API process
    and for video decoding. When inference runs on an accelerator (MPS, CUDA,
    CoreML, OpenVINO GPU) the CPU only does pre/post-processing, so a small
    budget is plenty.

    ``gpu_thread_cap`` is how small "plenty" is. The default of 2 suits the
    integrated accelerators, where the CPU is also the thing feeding them. A
    discrete CUDA card is different: inference leaves the cores free, but the
    worker still resizes, letterboxes, draws and JPEG-encodes every frame on
    them, so callers on that path pass a higher cap.
    """
    if configured > 0:
        return configured
    cores = physical_cores()
    streams = max(1, expected_streams)
    budget = max(1, cores // streams)
    if gpu:
        # The accelerator does the matmuls; more CPU threads only add contention.
        budget = min(budget, max(1, gpu_thread_cap))
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


def cpu_affinity_slice(slot: int, n_threads: int) -> list[int] | None:
    """Logical CPUs worker ``slot`` should be restricted to, or None.

    The ids are laid out so consecutive workers land on different physical
    cores, taking every SMT sibling of a core along with it.
    """
    logical = os.cpu_count() or 1
    cores = physical_cores()
    if cores <= 1 or n_threads <= 0:
        return None
    threads_per_core = max(1, logical // max(1, cores))
    total_slots = max(1, cores // n_threads)
    if total_slots <= 1:
        return None  # a single worker may as well have the whole machine
    first_core = (slot % total_slots) * n_threads
    picked: list[int] = []
    for core in range(first_core, min(first_core + n_threads, cores)):
        for sibling in range(threads_per_core):
            cpu = core * threads_per_core + sibling
            if cpu < logical:
                picked.append(cpu)
    return picked or None


def pin_worker_affinity(slot: int, n_threads: int, mode: str = "auto") -> list[int] | None:
    """Restrict this process to its own slice of the CPU.

    ``apply_worker_threads`` is not enough for every backend: the OpenVINO CPU
    plugin schedules on TBB and ignores ``OMP_NUM_THREADS``, so without this
    every worker spins up one thread per logical CPU. TBB *does* honour the
    process affinity mask, and so do OpenMP and torch, which makes this the one
    lever that works for all of them.

    Returns the CPU list that was applied, or None when nothing changed.
    """
    if mode == "off" or is_macos():
        # macOS has no per-process affinity API (and its scheduler moves work
        # between P and E cores on purpose).
        return None
    cpus = cpu_affinity_slice(slot, n_threads)
    if not cpus:
        return None
    try:
        import psutil

        psutil.Process().cpu_affinity(cpus)
        return cpus
    except Exception:
        return None


def describe() -> dict:
    """Small summary used by the /api/health endpoint."""
    info = {
        "platform": sys.platform,
        "machine": platform.machine(),
        "cpu": chip_name(),
        "vendor": cpu_vendor(),
        "cpu_cores": os.cpu_count(),
        "physical_cores": physical_cores(),
        # Kept under the old name too: the Apple Silicon docs refer to it.
        "performance_cores": physical_cores(),
        "apple_silicon": is_apple_silicon(),
        "gpu_vendor": gpu_vendor(),
        "openvino": openvino_version(),
        "openvino_devices": list(openvino_devices()),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        # None on a CPU-only wheel -- the quickest way to tell "no GPU here"
        # apart from "GPU present but the wrong torch build is installed".
        info["cuda_version"] = torch.version.cuda
        info["cuda_devices"] = [dict(d) for d in cuda_devices()]
        info["mps_available"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:
        info["torch"] = None
    return info
