"""Video source abstraction: resolves a source URL to a readable stream and
yields frames, with automatic reconnection on failure.

Supported types:
- youtube : resolved to a direct stream URL via yt-dlp
- rtsp / rtmp / hls / http / file : opened directly with OpenCV (FFmpeg backend)
"""
from __future__ import annotations

import time
from collections.abc import Iterator

import cv2


class SourceOpenError(RuntimeError):
    pass


def resolve_stream_url(source_type: str, url: str) -> str:
    """Return a URL that OpenCV/FFmpeg can open directly."""
    if source_type == "youtube":
        return _resolve_youtube(url)
    return url


# A single-stream selector (no "video+audio" merges, since OpenCV/FFmpeg opens
# one URL). Prefers HLS for live, then a muxed progressive <=720p, then a
# video-only stream (audio is irrelevant for detection), then anything.
_YT_FORMAT = (
    "best[protocol=m3u8_native]/"
    "best[vcodec!=none][acodec!=none][height<=720]/"
    "bestvideo[height<=720][protocol^=http]/"
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


def _resolve_youtube(url: str) -> str:
    """Resolve a YouTube (VOD or live) URL to a direct stream URL that
    OpenCV/FFmpeg can open. Tries several player clients for resilience."""
    import yt_dlp

    last_err: Exception | None = None
    for client in _YT_CLIENTS:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "format": _YT_FORMAT,
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


class SourceReader:
    """Iterates frames from a source, reconnecting on read failures."""

    def __init__(self, source_type: str, url: str, reconnect_delay: float = 3.0):
        self.source_type = source_type
        self.url = url
        self.reconnect_delay = reconnect_delay
        self._cap: cv2.VideoCapture | None = None

    def _open(self) -> cv2.VideoCapture:
        stream_url = resolve_stream_url(self.source_type, self.url)
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        # Keep a small buffer so we stay close to live for network streams.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            raise SourceOpenError(f"Failed to open source: {self.url}")
        return cap

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
