"""Video source abstraction: resolves a source URL to a readable stream and
yields frames, with automatic reconnection on failure.

Supported types:
- youtube : resolved to a direct stream URL via yt-dlp
- rtsp / rtmp / hls / http / file : opened directly with OpenCV (FFmpeg backend)
"""
from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Iterator

import cv2


class SourceOpenError(RuntimeError):
    pass


def resolve_stream_url(source_type: str, url: str, max_height: int = 720) -> str:
    """Return a URL that OpenCV/FFmpeg can open directly."""
    if source_type == "youtube":
        return _resolve_youtube(url, max_height)
    return url


# A single-stream selector (no "video+audio" merges, since OpenCV/FFmpeg opens
# one URL). Prefers HLS for live, then a muxed progressive <=720p, then a
# video-only stream (audio is irrelevant for detection), then anything.
def _yt_format(max_height: int) -> str:
    h = max(240, int(max_height or 720))
    return (
        f"best[protocol=m3u8_native][height<={h}]/"
        f"best[vcodec!=none][acodec!=none][height<={h}]/"
        f"bestvideo[height<={h}][protocol^=http]/"
        "best[protocol=m3u8_native]/"
        "bestvideo[protocol^=http]/best"
    )

# YouTube frequently breaks the default web client; trying alternate player
# clients is the standard yt-dlp workaround for "No video formats found".
_YT_CLIENTS = (None, ["android"], ["ios"], ["tv"], ["web_safari"], ["mweb"])


def _pick_stream_url(info: dict) -> str | None:
    if not info:
        return None
    if info.get("url"):
        return info["url"]
    for f in info.get("requested_formats") or []:
        if f.get("vcodec") not in (None, "none") and f.get("url"):
            return f["url"]
    # Fall back to the best video-bearing format from the full list.
    vids = [
        f
        for f in (info.get("formats") or [])
        if f.get("vcodec") not in (None, "none") and f.get("url")
    ]
    if vids:
        return vids[-1]["url"]
    return None


def _resolve_youtube(url: str, max_height: int = 720) -> str:
    """Resolve a YouTube (VOD or live) URL to a direct stream URL that
    OpenCV/FFmpeg can open. Tries several player clients for resilience."""
    import yt_dlp

    last_err: Exception | None = None
    for client in _YT_CLIENTS:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "format": _yt_format(max_height),
        }
        if client is not None:
            opts["extractor_args"] = {"youtube": {"player_client": client}}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                stream_url = _pick_stream_url(info)
                if stream_url:
                    return stream_url
        except Exception as exc:  # try the next player client
            last_err = exc
            continue

    hint = (
        "No playable format found via yt-dlp. The video may be private, "
        "age/geo-restricted, or yt-dlp needs updating "
        "(`pip install -U yt-dlp`)."
    )
    raise SourceOpenError(f"{hint} Last error: {last_err}")


# OpenCV reads FFmpeg options from this environment variable at the moment a
# VideoCapture is constructed, so it has to be set per open() call.
_CAPTURE_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"


# Protocols that push frames in real time: the sender sets the pace, so being
# behind means growing latency and the right response is to drop frames.
# Socket timeout for real-time protocols, in microseconds (FFmpeg's unit).
_SOCKET_TIMEOUT_US = 5_000_000

REALTIME_SOURCE_TYPES = ("rtsp", "rtmp")
# Protocols delivered in chunks over HTTP: FFmpeg downloads a whole segment
# (typically 2-6 seconds) as fast as the network allows, then waits for the
# next one. Frames therefore arrive in bursts and need a jitter buffer, not
# frame dropping — dropping would leave one frame per burst.
CHUNKED_SOURCE_TYPES = ("hls", "http", "youtube")
LIVE_SOURCE_TYPES = REALTIME_SOURCE_TYPES + CHUNKED_SOURCE_TYPES


# Hardware decoders OpenCV can select itself through CAP_PROP_HW_ACCELERATION.
# This is the documented API; the "hwaccel" key below is an FFmpeg *CLI* option
# and is not understood by avformat_open_input, so anything routed through the
# capture-options string relies on FFmpeg picking the decoder on its own.
_SOFTWARE_MODES = ("", "off", "none", "software")

# Our setting name -> the cv2.VIDEO_ACCELERATION_* suffix it maps to. Resolved
# with getattr rather than up front: builds differ in which types they expose
# (VIDEO_ACCELERATION_DRM only appeared in OpenCV 4.12), and a missing one must
# degrade instead of raising in the middle of a worker.
_ACCELERATION_TYPES = {
    # "Prefer hardware, fall back to software" -- D3D11VA on Windows,
    # VAAPI on Linux, VideoToolbox on macOS.
    "auto": "ANY",
    "any": "ANY",
    "d3d11va": "D3D11",
    "d3d11": "D3D11",
    "dxva2": "D3D11",
    # Intel Quick Sync, via the Media SDK / oneVPL path.
    "qsv": "MFX",
    "mfx": "MFX",
    "vaapi": "VAAPI",
    "drm": "DRM",
}


def _hw_acceleration(hwaccel: str) -> int | None:
    """Map our FFMPEG_HWACCEL setting onto an OpenCV acceleration constant.

    Returns None for values this OpenCV build cannot express as an acceleration
    type -- either because they are not one ("videotoolbox", "cuda") or because
    the build predates it. Those keep going through the FFmpeg option string.
    """
    mode = (hwaccel or "").strip().lower()
    if mode in _SOFTWARE_MODES:
        return getattr(cv2, "VIDEO_ACCELERATION_NONE", None)
    name = _ACCELERATION_TYPES.get(mode)
    if name is None:
        return None
    return getattr(cv2, f"VIDEO_ACCELERATION_{name}", None)


def _capture_options(source_type: str, hwaccel: str, rtsp_tcp: bool) -> str:
    """Build the ``key;value|key;value`` string OpenCV passes to FFmpeg.

    Only protocol/container level options belong here -- those are real
    AVFormat options. Hardware decoding is requested separately, through
    :func:`_hw_acceleration`, except for the backends OpenCV has no constant
    for (VideoToolbox, NVDEC).
    """
    opts: list[str] = []
    mode = (hwaccel or "").strip().lower()
    if mode not in _SOFTWARE_MODES and _hw_acceleration(hwaccel) is None:
        # e.g. "videotoolbox" on Apple Silicon: H.264/HEVC decoding moves to
        # the dedicated media engine, freeing the CPU cores for inference.
        opts.append(f"hwaccel;{hwaccel}")
    if rtsp_tcp and source_type == "rtsp":
        opts.append("rtsp_transport;tcp")
    if source_type in REALTIME_SOURCE_TYPES:
        # Don't let FFmpeg build up an internal buffer: on a live camera a full
        # buffer only means we are watching the past. Better to be a frame
        # behind than a second behind. (Not for HLS, which relies on buffering.)
        opts.append("fflags;nobuffer")
        opts.append("flags;low_delay")
        # A camera that drops off the network silently would otherwise block
        # the read forever; erroring out lets the reconnect loop take over.
        opts.append(f"timeout;{_SOCKET_TIMEOUT_US}")
    return "|".join(opts)


class SourceReader:
    """Iterates frames from a source, reconnecting on read failures."""

    def __init__(
        self,
        source_type: str,
        url: str,
        reconnect_delay: float = 3.0,
        hwaccel: str = "",
        rtsp_tcp: bool = True,
        youtube_max_height: int = 720,
    ):
        self.source_type = source_type
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.hwaccel = hwaccel
        self.rtsp_tcp = rtsp_tcp
        self.youtube_max_height = youtube_max_height
        # Frame rate reported by the stream, filled in on open.
        self.fps: float = 0.0
        # Set to False once hardware decoding has been proven not to work for
        # this stream, so we don't pay the failed-open cost on every reconnect.
        self._hwaccel_ok = hwaccel.strip().lower() not in _SOFTWARE_MODES
        self._cap: cv2.VideoCapture | None = None

    def _try_open(self, stream_url: str, hwaccel: str) -> cv2.VideoCapture | None:
        options = _capture_options(self.source_type, hwaccel, self.rtsp_tcp)
        accel = _hw_acceleration(hwaccel)
        params = [cv2.CAP_PROP_HW_ACCELERATION, accel] if accel is not None else []
        previous = os.environ.get(_CAPTURE_OPTIONS_ENV)
        if options:
            os.environ[_CAPTURE_OPTIONS_ENV] = options
        try:
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG, params)
        finally:
            if options:
                if previous is None:
                    os.environ.pop(_CAPTURE_OPTIONS_ENV, None)
                else:
                    os.environ[_CAPTURE_OPTIONS_ENV] = previous
        # Keep a small buffer so we stay close to live for network streams.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _open(self) -> cv2.VideoCapture:
        stream_url = resolve_stream_url(self.source_type, self.url, self.youtube_max_height)
        if self._hwaccel_ok:
            cap = self._try_open(stream_url, self.hwaccel)
            if cap is not None:
                self._read_fps(cap)
                return cap
            # Not every codec/container can be decoded in hardware; retry in
            # software rather than reporting the source as broken.
            print(
                f"[SourceReader] hwaccel '{self.hwaccel}' failed for "
                f"{self.source_type} source; using software decoding"
            )
            self._hwaccel_ok = False
        cap = self._try_open(stream_url, "")
        if cap is None:
            raise SourceOpenError(f"Failed to open source: {self.url}")
        self._read_fps(cap)
        return cap

    def _read_fps(self, cap: cv2.VideoCapture) -> None:
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        except Exception:
            fps = 0.0
        # Some streams report nonsense (0, or 90000 from a raw timebase).
        self.fps = fps if 1.0 < fps < 121.0 else 0.0

    def frames(self, stop_flag, on_error=None) -> Iterator["cv2.typing.MatLike"]:
        """Yield frames until ``stop_flag`` (a callable returning bool) is set.

        Reconnects automatically. ``on_error(message)`` is called whenever the
        source cannot be opened, so the worker can surface the reason (e.g. a
        yt-dlp extraction failure) to the dashboard instead of failing silently.
        """
        while not stop_flag():
            if self._cap is None:
                try:
                    self._cap = self._open()
                except Exception as exc:
                    if on_error:
                        on_error(str(exc))
                    if self.source_type == "file":
                        return  # a missing/unreadable file won't fix itself
                    time.sleep(self.reconnect_delay)
                    continue
            try:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    # End of file or transient network drop -> reconnect.
                    self._release()
                    if self.source_type == "file":
                        # A local file simply ended.
                        return
                    time.sleep(self.reconnect_delay)
                    continue
                yield frame
            except Exception:
                self._release()
                time.sleep(self.reconnect_delay)
        self._release()

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None




# How long the decoder waits for buffer space before deciding we are behind.
_BACKPRESSURE_TIMEOUT_SEC = 1.0


class BufferedFrameReader:
    """Decodes in a background thread so the worker never stalls the decoder.

    Two behaviours, because the two kinds of live source fail differently:

    * **Real-time push** (RTSP/RTMP) — the camera sets the pace. Falling behind
      means the decoder's queue grows and the dashboard drifts into the past in
      bursts. ``buffer_frames=1`` keeps only the newest frame and drops the
      rest, so latency stays flat under load.

    * **Chunked HTTP** (HLS / YouTube) — FFmpeg downloads a whole segment as
      fast as it can, then waits for the next. Frames arrive in bursts of
      hundreds followed by seconds of nothing. Here dropping is exactly wrong;
      instead the burst is buffered and released at the stream's own frame
      rate (``paced=True``), which is what a video player does.

    In both cases the oldest frame is dropped when the buffer is full: on a
    live source, being behind is worse than missing a frame.
    """

    def __init__(self, reader: SourceReader, buffer_frames: int = 1, paced: bool = False,
                 backpressure: bool = False, default_fps: float = 25.0):
        self._reader = reader
        self._paced = paced
        self._backpressure = backpressure
        self._default_fps = default_fps
        # A Queue, not a deque + Event: its condition variable has no
        # lost-wakeup race, which otherwise leaves the consumer blocked while
        # frames are already waiting for it.
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, buffer_frames))
        self.dropped = 0
        self.captured = 0

    @property
    def _interval(self) -> float:
        fps = self._reader.fps or self._default_fps
        return 1.0 / fps

    def _offer(self, frame) -> None:
        """Hand a frame to the consumer, dropping the oldest when we're behind.

        With ``backpressure`` the producer *waits* for room instead of spinning
        on a full buffer. That matters more than it looks: a chunked source is
        downloaded as fast as the network allows, and a decoder thread running
        flat out starves the detection loop of the GIL — the loop then runs at
        a fraction of the frame rate even though the GPU is nearly idle.
        Blocking puts the decoder to sleep and hands the time back.
        """
        if self._backpressure:
            try:
                self._queue.put(frame, timeout=_BACKPRESSURE_TIMEOUT_SEC)
                return
            except queue.Full:
                # Sustained overload: we cannot keep up with real time, so make
                # room by throwing away the oldest frame.
                pass
        while True:
            try:
                self._queue.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def frames(self, stop_flag, on_error=None) -> Iterator["cv2.typing.MatLike"]:
        finished = threading.Event()

        def pump() -> None:
            try:
                for frame in self._reader.frames(stop_flag, on_error=on_error):
                    self.captured += 1
                    self._offer(frame)
            finally:
                finished.set()
                # Unblock a consumer that is waiting on an empty queue.
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    pass

        thread = threading.Thread(target=pump, name="source-reader", daemon=True)
        thread.start()
        next_due = time.monotonic()
        try:
            while not stop_flag():
                try:
                    frame = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if finished.is_set():
                        break
                    next_due = time.monotonic()
                    continue
                if frame is None:  # producer finished
                    break

                if self._paced:
                    # Release on the stream's own clock so a downloaded burst
                    # plays back as smooth video rather than a jump forward.
                    now = time.monotonic()
                    if next_due > now:
                        time.sleep(min(next_due - now, 1.0))
                    # Don't let the schedule run away if we fell behind.
                    next_due = max(next_due, now - self._interval) + self._interval
                yield frame
        finally:
            thread.join(timeout=5.0)
