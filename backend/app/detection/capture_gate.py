"""When a vehicle in the ANPR zone is a new passage, and when it is one we have.

A capture answers "did a vehicle come past here", so it must fire once per
passage. :class:`~app.detection.vehicle_registry.VehicleRegistry` already tries
to guarantee that by holding the ``captured`` flag against the vehicle rather
than the tracker id -- but it can only do so for as long as it recognises the
vehicle, and on a zoomed-in overview camera it frequently cannot.

Two ways that fails, both measured on this project's own recordings (324 rows
from two 4K market cameras; 59 of them landed within five seconds of a row of
the same class from the same camera):

**The tracker renumbers a vehicle that has moved too far to re-identify.** The
registry adopts a new track id onto a known vehicle when the boxes overlap by
``ALPR_REID_GAP_SECONDS`` / IoU 0.6. At 4K with a second between detections, a
motorcycle crosses more than its own box length: two captures of one bike (the
plate read ``B 6984 NK`` in both) scored **IoU 0.17**, and another pair scored
**0.00** -- their boxes did not touch at all. No position-based re-identification
reaches those, and lowering the threshold far enough to would start merging
vehicles that really are different.

**The detector reports one vehicle as two nested boxes.** A motorcycle and its
rider come back both as one box around the pair and as a second box around just
the machine. Those arrive in the *same frame*, so there is no renumbering to
undo and no gap to re-identify across; they simply score IoU 0.37, under the
threshold, while one box sits entirely inside the other.

So this module stops asking which vehicle a box belongs to, and asks the two
questions that survive an unreliable tracker:

* **Is the vehicle all the way inside the zone?** Every duplicate pair measured
  had a first row 94.8%-98.8% inside the zone -- a bike clipped by the bottom
  edge of the frame on its way out -- and a second row at exactly 100%. A zone
  is a gate, and a vehicle straddling it is not through it yet. Waiting for the
  whole box costs nothing but a few frames, removes that entire class of
  duplicate, and leaves the better evidence: the crop that shows the whole
  machine rather than the half of it still in frame.

* **Have we just recorded a box that this one is inside?** Nested detections of
  one vehicle overlap almost totally on the smaller of the two boxes, which is
  the measure IoU is bad at. A short per-class ledger of what was recently
  captured catches them without any notion of identity at all.

Deliberately stdlib-only and free of any frame, model or database dependency,
for the same reason ``vehicle_registry`` is: this is geometry and a short list,
and it is worth being able to test as one.
"""
from __future__ import annotations

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: Box, b: Box) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    return iw * ih if iw > 0 and ih > 0 else 0.0


def containment(box: Box, other: Box) -> float:
    """How completely ``box`` and ``other`` cover each other, on the smaller one.

    1.0 when either lies wholly inside the other, 0.0 when they do not meet.

    Measuring against the smaller of the two is what makes this usable both for
    "is the vehicle inside the zone" and for "is this box the one we just
    captured". Against the zone it normally reads as the fraction of the vehicle
    inside it, because a vehicle is smaller than the zone it is driving through;
    but a lorry that fills the frame, or a zone drawn tight around a gate, would
    otherwise be permanently incapable of being wholly inside and so could never
    be captured at all. Turning the question around for that case -- does the
    vehicle cover the whole zone -- keeps the rule meaningful at both extremes.
    """
    smaller = min(_area(box), _area(other))
    if smaller <= 0:
        return 0.0
    return _intersection(box, other) / smaller


class CaptureGate:
    """The zone as a gate: one capture per vehicle that passes through it.

    ``min_containment`` is how much of the vehicle has to be inside the zone
    before it counts as having arrived. 1.0 is the literal reading -- every
    pixel of the box -- and the default sits just under it so that a pixel of
    detector jitter at the zone edge does not reject a vehicle plainly inside.
    There is more room there than it looks: the clipped first rows of the
    duplicated pairs scored 0.9485 to 0.9859, while a box overhanging the edge
    by one pixel scores 0.9978. Lower it towards ``ALPR_ZONE_MIN_OVERLAP`` and
    the gate widens back into a region, with the duplicates that come with it.

    Nothing here is a one-shot judgement: the gate is asked again on every frame
    of the passage, so being strict costs a frame or two of latency rather than
    the capture. What it does cost is a vehicle that never fits wholly inside
    the zone at all -- so the zone wants drawing with room for the largest
    vehicle worth recording.

    ``window`` and ``min_overlap`` govern the ledger: for ``window`` seconds
    after a capture, another box of the same class that overlaps it by
    ``min_overlap`` -- measured on the smaller box, so a nested detection scores
    1.0 -- is taken to be the same vehicle again rather than a new one. The
    window is deliberately short. It is there to span the frame or two in which
    one vehicle is reported twice, not to police a queue of traffic, and every
    second of it is a second in which a genuinely different vehicle arriving in
    the same place would go unrecorded.
    """

    def __init__(
        self,
        min_containment: float = 0.99,
        window: float = 3.0,
        min_overlap: float = 0.8,
    ):
        self.min_containment = min(1.0, max(0.0, min_containment))
        self.window = max(0.0, window)
        self.min_overlap = min(1.0, max(0.0, min_overlap))
        # class name -> [(box, when)], most recent last. Bounded by ``window``
        # rather than by count: what matters is how long ago a capture was, and
        # a busy junction is exactly when the list must not be truncated.
        self._recent: dict[str, list[tuple[Box, float]]] = {}

    # ------------------------------------------------------------------
    def is_through(self, box: Box, zone: Box) -> bool:
        """Is the vehicle far enough inside the zone to count as through it?"""
        return containment(box, zone) >= self.min_containment

    def is_repeat(self, class_name: str, box: Box, now: float) -> bool:
        """Does this box look like one we captured a moment ago?"""
        if self.window <= 0:
            return False
        self._forget(class_name, now)
        return any(
            containment(box, seen) >= self.min_overlap
            for seen, _when in self._recent.get(class_name, ())
        )

    def record(self, class_name: str, box: Box, now: float) -> None:
        """Remember a capture that actually happened.

        Kept separate from :meth:`is_repeat` because a capture is offered to a
        queue that is allowed to refuse it: recording one that was dropped would
        suppress the retry, and the vehicle would go unrecorded entirely.
        """
        if self.window <= 0:
            return
        self._forget(class_name, now)
        self._recent.setdefault(class_name, []).append((box, now))

    # ------------------------------------------------------------------
    def _forget(self, class_name: str, now: float) -> None:
        seen = self._recent.get(class_name)
        if not seen:
            return
        fresh = [(b, t) for (b, t) in seen if (now - t) <= self.window]
        if fresh:
            self._recent[class_name] = fresh
        else:
            self._recent.pop(class_name, None)
