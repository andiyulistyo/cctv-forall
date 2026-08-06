"""Line-crossing counter — the core value of this project.

A single counting line is defined by two points A and B (normalized 0..1).
For every tracked object we remember its previous centroid. A crossing is
registered when the segment traced by the object between two consecutive
frames (prev -> curr) actually intersects the line segment A-B. This is far
more robust than only testing the sign of a point against an infinite line:
it ignores objects that pass the line's extension outside the drawn segment,
and it naturally debounces jitter because a second crossing requires the
object to move back across and then over the line again.

Direction ("in" vs "out") is derived from which side the object ended up on,
which the caller maps to human labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _cross(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    """Cross product of (B-A) x (P-A). Sign = side of point P vs line A->B."""
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Return True if segment p1p2 intersects segment p3p4."""

    def orient(a, b, c) -> float:
        return _cross(a[0], a[1], b[0], b[1], c[0], c[1])

    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)

    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True

    # Collinear/touching cases are treated as non-crossings on purpose to
    # avoid double counting when an object slides along the line.
    return False


@dataclass
class _Track:
    last_point: tuple[float, float]
    last_seen: int


@dataclass
class Crossing:
    track_id: int
    class_name: str
    direction: str  # "in" | "out"


@dataclass
class LineCounter:
    """Maintains per-object state and cumulative per-class/direction counts.

    Coordinates are in pixel space; the counting line is provided normalized
    and scaled to the current frame size via :meth:`set_frame_size`.
    """

    line_norm: tuple[tuple[float, float], tuple[float, float]]
    # cumulative counts: {class_name: {"in": n, "out": n}}
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    _tracks: dict[int, _Track] = field(default_factory=dict)
    _frame_idx: int = 0
    _a_px: tuple[float, float] = (0.0, 0.0)
    _b_px: tuple[float, float] = (0.0, 0.0)
    _stale_frames: int = 90  # drop a track unseen for this many processed frames

    def set_frame_size(self, width: int, height: int) -> None:
        (ax, ay), (bx, by) = self.line_norm
        self._a_px = (ax * width, ay * height)
        self._b_px = (bx * width, by * height)

    def _bump(self, class_name: str, direction: str) -> None:
        bucket = self.counts.setdefault(class_name, {"in": 0, "out": 0})
        bucket[direction] += 1

    def update(self, detections: list[tuple[int, str, float, float]]) -> list[Crossing]:
        """Feed current-frame detections and return the crossings just made.

        Each detection is ``(track_id, class_name, cx, cy)`` where (cx, cy) is
        the object centroid in pixels.
        """
        self._frame_idx += 1
        crossings: list[Crossing] = []
        a, b = self._a_px, self._b_px

        for track_id, class_name, cx, cy in detections:
            curr = (cx, cy)
            prev_track = self._tracks.get(track_id)
            if prev_track is not None:
                prev = prev_track.last_point
                if _segments_intersect(prev, curr, a, b):
                    side = _cross(a[0], a[1], b[0], b[1], cx, cy)
                    direction = "in" if side > 0 else "out"
                    self._bump(class_name, direction)
                    crossings.append(Crossing(track_id, class_name, direction))
            self._tracks[track_id] = _Track(last_point=curr, last_seen=self._frame_idx)

        self._prune()
        return crossings

    def _prune(self) -> None:
        cutoff = self._frame_idx - self._stale_frames
        stale = [tid for tid, t in self._tracks.items() if t.last_seen < cutoff]
        for tid in stale:
            del self._tracks[tid]

    def total(self) -> dict:
        return {cls: dict(dirs) for cls, dirs in self.counts.items()}
