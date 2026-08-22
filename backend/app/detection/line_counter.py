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

A bare intersection test is not enough on its own, though. Detection boxes
jitter, so an object sitting near the line can be nudged back and forth across
it and be counted every time -- on the sample junction feed that produced 8.5%
more crossings than there were cars and 18.5% more than there were trucks, in
the tell-tale ``out,in,out`` pattern. So a side only counts as *reached* once
the object is clear of the line by a margin (``hysteresis``); inside that band
its side is simply undecided and nothing is registered.
"""
from __future__ import annotations

import math
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
    # -1 / +1 for the last side the object was decidedly on, 0 while it has
    # never been clear of the line, and where it was when that was decided.
    # The crossing test runs between two *decided* positions, which straddle
    # the line by at least the margin and so cannot be jitter.
    side: int = 0
    side_point: tuple[float, float] | None = None


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
    # How far from the line, as a fraction of frame height, an object must get
    # before that side counts as reached. 0 disables the band entirely and
    # restores the raw intersection test.
    hysteresis: float = 0.02
    # cumulative counts: {class_name: {"in": n, "out": n}}
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    _tracks: dict[int, _Track] = field(default_factory=dict)
    _frame_idx: int = 0
    _a_px: tuple[float, float] = (0.0, 0.0)
    _b_px: tuple[float, float] = (0.0, 0.0)
    _margin_px: float = 0.0
    _line_len: float = 0.0
    _stale_frames: int = 90  # drop a track unseen for this many processed frames

    def set_frame_size(self, width: int, height: int) -> None:
        (ax, ay), (bx, by) = self.line_norm
        self._a_px = (ax * width, ay * height)
        self._b_px = (bx * width, by * height)
        # _cross returns twice the triangle area, so dividing by the line
        # length turns it into a perpendicular distance in pixels.
        self._line_len = math.hypot(
            self._b_px[0] - self._a_px[0], self._b_px[1] - self._a_px[1]
        )
        self._margin_px = max(0.0, self.hysteresis) * height

    def _side_of(self, cx: float, cy: float) -> int:
        """-1 / +1 for a point clear of the line, 0 while inside the margin."""
        (ax, ay), (bx, by) = self._a_px, self._b_px
        signed = _cross(ax, ay, bx, by, cx, cy)
        if self._line_len <= 0.0:
            return 0
        distance = signed / self._line_len
        if abs(distance) < self._margin_px:
            return 0
        return 1 if distance > 0 else -1

    def _bump(self, class_name: str, direction: str) -> None:
        bucket = self.counts.setdefault(class_name, {"in": 0, "out": 0})
        bucket[direction] += 1

    def update(self, detections: list[tuple[int, str, float, float]]) -> list[Crossing]:
        """Feed current-frame detections and return the crossings just made.

        Each detection is ``(track_id, class_name, cx, cy)`` where (cx, cy) is
        the object's ground contact point in pixels -- see
        ``Detection.ground_point`` for why that and not the centroid.
        """
        self._frame_idx += 1
        crossings: list[Crossing] = []
        a, b = self._a_px, self._b_px

        for track_id, class_name, cx, cy in detections:
            curr = (cx, cy)
            side = self._side_of(cx, cy)
            track = self._tracks.get(track_id)
            if track is None:
                self._tracks[track_id] = _Track(
                    last_point=curr,
                    last_seen=self._frame_idx,
                    side=side,
                    side_point=curr if side else None,
                )
                continue

            if side != 0:
                changed_sides = track.side != 0 and side != track.side
                # The side change says the object got clear of the line on the
                # other side; the intersection test says it did so *across the
                # drawn segment* rather than past its extension.
                if (
                    changed_sides
                    and track.side_point is not None
                    and _segments_intersect(track.side_point, curr, a, b)
                ):
                    direction = "in" if side > 0 else "out"
                    self._bump(class_name, direction)
                    crossings.append(Crossing(track_id, class_name, direction))
                track.side = side
                track.side_point = curr

            track.last_point = curr
            track.last_seen = self._frame_idx

        self._prune()
        return crossings

    def _prune(self) -> None:
        cutoff = self._frame_idx - self._stale_frames
        stale = [tid for tid, t in self._tracks.items() if t.last_seen < cutoff]
        for tid in stale:
            del self._tracks[tid]

    def total(self) -> dict:
        return {cls: dict(dirs) for cls, dirs in self.counts.items()}
