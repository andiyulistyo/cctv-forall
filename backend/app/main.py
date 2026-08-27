"""FastAPI application entrypoint."""
from __future__ import annotations

import signal
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, update

from .auth import ensure_admin_user
from .config import settings
from .database import SessionLocal, init_db
from .detection.manager import SHUTTING_DOWN, get_manager, init_manager
from .models import Source
from .retention import start_scheduler, stop_scheduler
from . import runtime
from .api import auth_routes, counts, faces, plates, sources, streams

FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


def _install_shutdown_signal_handlers() -> None:
    """Flip SHUTTING_DOWN the moment Ctrl-C arrives.

    The MJPEG endpoints are endless generators that only stop once that event
    is set, and uvicorn runs the lifespan shutdown -- where the manager sets it
    -- *after* every in-flight response has finished. Each side ends up waiting
    for the other, which is why Ctrl-C used to hang on "Waiting for connections
    to close". Set it from the signal itself, then hand over to whoever was
    handling the signal already (uvicorn) so the rest of shutdown is unchanged.
    """
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous = signal.getsignal(sig)
        except (ValueError, OSError):
            continue
        # No existing handler means nothing is arranging a graceful stop; leave
        # the default in place rather than swallowing the signal.
        if not callable(previous):
            continue

        def handler(signum, frame, _previous=previous):
            SHUTTING_DOWN.set()
            _previous(signum, frame)

        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # not the main thread
            continue


# Seconds between two auto-started workers. Spawning one is a fresh
# interpreter that loads torch and the model; releasing four at the same
# instant on a machine that has only just booted makes all four slower.
AUTO_START_STAGGER_SECONDS = 3.0


def _resume_auto_start_sources() -> threading.Thread | None:
    """Start the sources marked ``auto_start``, off the startup path.

    Spawning a worker costs a second or two, so doing this inline would keep
    the dashboard unreachable for as long as all of them together take --
    exactly when someone is reloading the page to see whether the machine came
    back. The thread is a daemon and checks SHUTTING_DOWN between starts, so a
    Ctrl-C part-way through a slow resume still exits.

    Returns the thread (None if nothing is marked) so tests can wait for it.
    """
    db = SessionLocal()
    try:
        pending = [
            (src.id, sources.build_source_cfg(src))
            for src in db.scalars(
                select(Source).where(Source.auto_start == 1).order_by(Source.id)
            )
        ]
    finally:
        db.close()
    if not pending:
        return None

    def _mark(source_id: int, status: str, message: str | None = None) -> None:
        """Write a status back, swallowing whatever the write hits.

        A locked database is not a reason to abandon the sources further down
        the list: the row is only what the dashboard displays, while the worker
        it describes is already up.
        """
        db = SessionLocal()
        try:
            db.execute(
                update(Source)
                .where(Source.id == source_id)
                .values(status=status, status_message=message)
            )
            db.commit()
        except Exception as exc:  # pragma: no cover - depends on the DB
            print(f"[auto-start] source {source_id}: status not saved ({exc})")
        finally:
            db.close()

    def _resume() -> None:
        mgr = get_manager()
        for index, (source_id, cfg) in enumerate(pending):
            # wait() returns True only if the event was set, i.e. we are
            # shutting down; a plain sleep would ignore that for 3 seconds.
            if index and SHUTTING_DOWN.wait(AUTO_START_STAGGER_SECONDS):
                return
            if SHUTTING_DOWN.is_set():
                return
            # One source that refuses to spawn must not strand the rest. There
            # is nobody at an unattended machine to start the others by hand --
            # that is the whole point of the flag -- so record the failure the
            # way a worker does and carry on down the list.
            try:
                mgr.start(cfg)
            except Exception as exc:
                print(f"[auto-start] source {source_id}: start failed ({exc})")
                _mark(source_id, "error", f"auto-start gagal: {exc}")
                continue
            _mark(source_id, "starting")

    thread = threading.Thread(target=_resume, name="auto-start", daemon=True)
    thread.start()
    return thread


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_admin_user()
    # Workers don't survive a restart -> mark everything stopped.
    db = SessionLocal()
    try:
        db.execute(update(Source).values(status="stopped", status_message=None))
        db.commit()
    finally:
        db.close()
    init_manager()
    # ...and then bring back the ones that are supposed to run unattended.
    _resume_auto_start_sources()
    start_scheduler()
    _install_shutdown_signal_handlers()
    try:
        yield
    finally:
        stop_scheduler()
        try:
            get_manager().shutdown()
        except Exception:
            pass


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    """Liveness plus the runtime facts you need to confirm acceleration is on
    (device actually selected, hardware decoder, thread budget per worker)."""
    from .detection.detector import gpu_thread_cap, plan_inference

    model_path = settings.yolo_model_path
    plan = plan_inference(model_path, settings.device, settings.inference_half)
    return {
        "status": "ok",
        "app": settings.app_name,
        "detection": {
            "model": model_path,
            "device": plan.device,
            "backend": plan.backend,
            "imgsz": settings.inference_imgsz,
            "half": plan.half,
            "frame_stride": settings.frame_stride,
            "process_width": settings.process_width or None,
            "threads_per_worker": runtime.threads_per_worker(
                settings.threads_per_worker,
                settings.expected_streams,
                gpu=not plan.cpu_bound,
                gpu_thread_cap=gpu_thread_cap(plan),
            ),
            "ffmpeg_hwaccel": settings.ffmpeg_hwaccel or None,
        },
        "hardware": runtime.describe(),
    }


# Everything the API serves lives under /api, which keeps it out of the way of
# the SPA's own routes. Without the prefix, "/plates" and "/faces" were claimed
# by the API -- so opening or refreshing one of those pages in the browser
# answered with a 401 JSON body instead of the app, and a link to a filtered
# view could not be shared at all.
for _router in (
    auth_routes.router,
    sources.router,
    streams.router,
    counts.router,
    plates.router,
    faces.router,
):
    app.include_router(_router, prefix="/api")


@app.api_route("/api/{rest:path}", include_in_schema=False)
def unknown_api_route(rest: str):
    """A mistyped endpoint should say so.

    The SPA is mounted as a catch-all below, so without this an unknown /api
    path would fall through and return index.html with a cheerful 200 -- which
    reads like a broken frontend rather than a wrong URL.
    """
    raise HTTPException(404, f"No such API endpoint: /api/{rest}")


class SPAStaticFiles(StaticFiles):
    """Serve the built SPA, falling back to index.html for client-side routes."""

    async def get_response(self, path: str, scope):
        from starlette.exceptions import HTTPException as StarletteHTTPException

        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                index = Path(self.directory) / "index.html"
                if index.exists():
                    return FileResponse(str(index))
            raise


if FRONTEND_DIST.exists():
    app.mount("/", SPAStaticFiles(directory=str(FRONTEND_DIST), html=True), name="spa")
else:  # Frontend not built yet (e.g. dev via vite on :5173).

    @app.get("/")
    def root():
        return {
            "message": "Frontend not built. Run the Vite dev server or build it.",
            "docs": "/docs",
        }
