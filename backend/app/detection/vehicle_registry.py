"""Which physical vehicle a tracker id belongs to.

Track ids are not identities. ByteTrack retires a track as soon as it stops
matching detections and gives the same vehicle a fresh id when it picks it up
again -- which is exactly what keeps happening to a vehicle that is *not*
moving. A car parked in view sits at the edge of the detector's confidence,
flickers in and out over the course of a minute, and every flicker is a new id.

All of the "read this vehicle once" logic in ANPR hangs off that id: the
attempt budget, the "already read it well enough" check, and the database row a
better read corrects in place. Renumber the vehicle and every one of those
resets, so a parked car is read, and written, all over again -- the same car
appearing in the plate list every few minutes, each row carrying a different
(and equally wrong) OCR string.

This registry sits in front of that state and answers two questions:

* **Is this new track id a vehicle we already know?** A box that appears where
  one we were watching a moment ago sat, with no live track of its own, is that
  vehicle. It is adopted: same attempts, same best read, same row.
* **Has this vehicle stopped?** Movement is measured against the place it was
  last seen to move from rather than against the previous frame, so a slow
  creep still counts as motion while detector jitter does not. A vehicle that
  has not moved for ``stationary_seconds`` is parked -- it has already had its
  reads, and more of them describe nothing new.

Deliberately stdlib-only and free of any frame, model or database dependency:
this is a small state machine and is worth being able to test as one.
"""
from __future__ import annotations

from dataclasses import dataclass

Box = tuple[float, float, float, float]  # x1, y1, x2, y2

# Overlap demanded before a vehicle that has been out of sight for longer than
# ``reid_gap`` is recognised again. Higher than the ordinary threshold because
# the ordinary one leans on the gap being tiny: here the box has to be all but
# the same box, which is what a parked car that the detector lost and found
# again looks like.
_PARKED_REID_IOU = 0.75


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Vehicle:
    """One physical vehicle, as opposed to one tracker id.

    The plate fields live here rather than in per-track dictionaries precisely
    so they survive the tracker renumbering the vehicle. They are written by the
    ANPR thread and read by the detection loop; each is a single attribute
    assignment, so the worst a race can do is act on a value one frame old.
    """

    vid: int
    track_id: int
    box: Box
    first_seen: float
    last_seen: float
    # Where the vehicle was when it was last seen to move, and when that was.
    # Motion is judged against this, never against the previous frame: a vehicle
    # creeping forward a pixel per frame is moving, and comparing consecutive
    # frames would call it parked.
    anchor: Box = (0.0, 0.0, 0.0, 0.0)
    moved_at: float = 0.0
    # --- plate state, carried across renumbering ---
    attempts: int = 0
    best_conf: float = 0.0
    text: str = ""
    row_id: int | None = None
    # Already recorded once by the capture path (see ALPR_CAPTURE_CLASSES).
    # Carried across renumbering for the same reason the plate state is: a
    # motorcycle that stops in the zone is renumbered every few seconds, and a
    # per-track flag would record it again on each new id.
    captured: bool = False


class VehicleRegistry:
    """Stable vehicle identities on top of an unstable tracker.

    ``reid_gap`` bounds how long a vehicle may be missing before a box in the
    same place counts as a *different* vehicle. It is the entire safety margin
    of the adoption rule: at a gate, cars stop one behind another in the same
    spot, and the only thing separating "the tracker blinked" from "the next car
    pulled up" is that a blink is short and a changeover is not. Keep it at a
    few seconds -- long enough for the tracker to renumber, far too short for
    one car to leave and another to take its place.

    A vehicle already established as *parked* is the exception and gets
    ``parked_memory`` instead. The detector does not merely renumber a
    motionless car, it loses it outright -- a static shape against a static
    background drifts below the confidence threshold and stays there for
    minutes -- so a few seconds of memory would let the same parked car keep
    coming back as a new arrival, which is the whole complaint. What makes the
    longer window safe is what it would take to make it wrong: the parked car
    must leave and another take its place entirely between two detections,
    while a moving vehicle is the easiest thing in the frame to detect.
    """

    def __init__(
        self,
        reid_gap: float = 4.0,
        reid_iou: float = 0.6,
        stationary_seconds: float = 20.0,
        motion_fraction: float = 0.15,
        forget_seconds: float = 60.0,
        parked_memory: float = 300.0,
    ):
        self.reid_gap = max(0.0, reid_gap)
        self.reid_iou = reid_iou
        self.stationary_seconds = max(0.0, stationary_seconds)
        self.motion_fraction = motion_fraction
        self.forget_seconds = max(1.0, forget_seconds)
        self.parked_memory = max(0.0, parked_memory)
        self._vehicles: dict[int, Vehicle] = {}
        self._by_track: dict[int, Vehicle] = {}
        self._next_vid = 1

    # ------------------------------------------------------------------
    def observe(self, track_id: int, box: Box, now: float) -> Vehicle:
        """Record that ``track_id`` was seen at ``box``; return its vehicle."""
        vehicle = self._by_track.get(track_id)
        if vehicle is None:
            vehicle = self._adopt(track_id, box, now)
        if vehicle is None:
            vehicle = Vehicle(
                vid=self._next_vid,
                track_id=track_id,
                box=box,
                first_seen=now,
                last_seen=now,
                anchor=box,
                moved_at=now,
            )
            self._next_vid += 1
            self._vehicles[vehicle.vid] = vehicle
            self._by_track[track_id] = vehicle
            return vehicle
        self._update(vehicle, box, now)
        return vehicle

    def get(self, vid: int) -> Vehicle | None:
        return self._vehicles.get(vid)

    def is_parked(self, vehicle: Vehicle, now: float) -> bool:
        """Has this vehicle been standing still long enough to stop reading it?"""
        if self.stationary_seconds <= 0:
            return False
        return (now - vehicle.moved_at) >= self.stationary_seconds

    def labels(self) -> dict[int, str]:
        """track id -> plate text, for the live overlay."""
        return {tid: v.text for tid, v in self._by_track.items() if v.text}

    def prune(self, now: float) -> None:
        """Forget vehicles that have been out of sight for a while."""
        gone = [
            v for v in self._vehicles.values()
            if (now - v.last_seen) > self._memory_for(v)
        ]
        for v in gone:
            self._vehicles.pop(v.vid, None)
            if self._by_track.get(v.track_id) is v:
                self._by_track.pop(v.track_id, None)

    # ------------------------------------------------------------------
    def _was_parked(self, vehicle: Vehicle) -> bool:
        """Was this vehicle standing still when we last saw it?"""
        if self.stationary_seconds <= 0:
            return False
        return (vehicle.last_seen - vehicle.moved_at) >= self.stationary_seconds

    def _memory_for(self, vehicle: Vehicle) -> float:
        if self._was_parked(vehicle):
            return max(self.forget_seconds, self.parked_memory)
        return self.forget_seconds

    def _adopt(self, track_id: int, box: Box, now: float) -> Vehicle | None:
        """Find the vehicle this new track id is a renumbering of, if any."""
        if self.reid_gap <= 0:
            return None
        best: Vehicle | None = None
        best_score = 0.0
        for v in self._vehicles.values():
            # Already accounted for in this very frame: whatever that box is, it
            # is not also this one.
            if v.last_seen >= now:
                continue
            gap = now - v.last_seen
            long_gap = gap > self.reid_gap
            if long_gap and not (self._was_parked(v) and gap <= self.parked_memory):
                continue
            score = iou(v.box, box)
            # A long absence has to be paid for with a near-exact match.
            if score <= (_PARKED_REID_IOU if long_gap else self.reid_iou):
                continue
            if score > best_score:
                best, best_score = v, score
        if best is None:
            return None
        # Move the vehicle onto its new id. The old id is dead as far as the
        # tracker is concerned, and leaving it mapped would keep the overlay
        # labelling a track that no longer exists.
        if self._by_track.get(best.track_id) is best:
            self._by_track.pop(best.track_id, None)
        best.track_id = track_id
        self._by_track[track_id] = best
        return best

    def _update(self, vehicle: Vehicle, box: Box, now: float) -> None:
        ax1, ay1, ax2, ay2 = vehicle.anchor
        aw, ah = ax2 - ax1, ay2 - ay1
        span = max(aw, ah, 1.0)
        acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Two ways to move: across the frame, or towards/away from the camera --
        # a vehicle driving straight at the lens barely shifts its centre while
        # its box grows, and calling that parked would be wrong.
        drift = max(abs(cx - acx), abs(cy - acy))
        growth = abs((x2 - x1) - aw)
        if drift > self.motion_fraction * span or growth > self.motion_fraction * max(aw, 1.0):
            vehicle.anchor = box
            vehicle.moved_at = now
        vehicle.box = box
        vehicle.last_seen = now
