"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import update

from .auth import ensure_admin_user
from .config import settings
from .database import SessionLocal, init_db
from .detection.manager import get_manager, init_manager
from .models import Source
from .retention import start_scheduler, stop_scheduler
from . import runtime
from .api import auth_routes, counts, faces, plates, sources, streams

FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


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
    start_scheduler()
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
    from .detection.detector import is_non_torch_model, pick_device

    device = pick_device(settings.device)
    model_path = settings.yolo_model_path
    accelerated = device != "cpu" or is_non_torch_model(model_path)
    return {
        "status": "ok",
        "app": settings.app_name,
        "detection": {
            "model": model_path,
            "device": "coreml" if is_non_torch_model(model_path) else device,
            "imgsz": settings.inference_imgsz,
            "half": settings.inference_half and device != "cpu",
            "frame_stride": settings.frame_stride,
            "process_width": settings.process_width or None,
            "threads_per_worker": runtime.threads_per_worker(
                settings.threads_per_worker, settings.expected_streams, gpu=accelerated
            ),
            "ffmpeg_hwaccel": settings.ffmpeg_hwaccel or None,
        },
        "hardware": runtime.describe(),
    }


app.include_router(auth_routes.router)
app.include_router(sources.router)
app.include_router(streams.router)
app.include_router(counts.router)
app.include_router(plates.router)
app.include_router(faces.router)


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
