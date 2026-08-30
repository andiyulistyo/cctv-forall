"""Consensus over every reading of one vehicle's plate.

A vehicle is read many times as it crosses the zone -- up to
``ALPR_MAX_ATTEMPTS`` of them -- and each reading is a guess at the *same*
string. Keeping only the most confident guess throws the rest away, and the
rest is where the answer is: OCR confidence is a statement about how cleanly
the decoder resolved that one image, not about how likely the string is to be
the plate. Eight frames reading "B1234XYZ" at 0.70 and one reading "B1234XY2"
at 0.72 is not a tie the 0.72 should win, and under a plain argmax it does.

So the readings vote, per character, weighted by the confidence they came with.

Alignment is the only hard part. Plate strings differ in length between
readings -- a frame that missed the leading area letter turns "B1234XYZ" into
"1234XYZ", and voting position-by-position across those two would put "1"
against "B" and score every column wrong. Readings are therefore grouped by
*shape* first (how many area letters, digits and suffix letters) and only
readings of the same shape are ever compared column by column, because within
one shape the columns are guaranteed to mean the same thing. The shape with the
most confidence behind it wins, and the vote runs inside it.

Deliberately stdlib-only, like ``vehicle_registry``: this is a small piece of
arithmetic over strings, and it is worth being able to test as one, without
OpenCV, a model, or a frame.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

# The three blocks of an Indonesian plate. Mirrors alpr._PLATE_RE, which every
# text reaching this module has already been through -- but kept as its own
# expression, with groups, because what is needed here is the split and not the
# yes/no.
_SHAPE_RE = re.compile(r"^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$")

# Readings kept per vehicle. The attempt budget already bounds this to a couple
# of dozen; the cap is only here so a vehicle that somehow never leaves the
# frame cannot grow the list without limit.
_MAX_READINGS = 64


@dataclass(frozen=True)
class _Reading:
    text: str
    conf: float


def plate_shape(text: str) -> tuple[int, int, int]:
    """The (area, digits, suffix) block lengths of a plate string.

    Anything that is not plate-shaped gets ``(-1, len, -1)``, which still keeps
    strings of one length together -- their columns line up even when the
    blocks cannot be named -- while never colliding with a real shape.
    """
    m = _SHAPE_RE.match(text)
    if m is None:
        return (-1, len(text), -1)
    return (len(m.group(1)), len(m.group(2)), len(m.group(3)))


@dataclass
class PlateVoter:
    """Every reading of one vehicle's plate, and the consensus over them.

    Lives on :class:`~app.detection.vehicle_registry.Vehicle` so it survives the
    tracker renumbering the vehicle -- the same reason the rest of the plate
    state lives there. Written only by the ANPR thread.
    """

    readings: list[_Reading] = field(default_factory=list)
    # Cached consensus, recomputed on add() rather than on read: the detection
    # loop asks for the text far more often than the ANPR thread produces one.
    text: str = ""
    confidence: float = 0.0

    def add(self, text: str, conf: float) -> bool:
        """Fold in one reading. Returns True if the consensus changed.

        A caller that gets False has nothing new to write: the vote landed
        where it already was.
        """
        if not text:
            return False
        if len(self.readings) < _MAX_READINGS:
            self.readings.append(_Reading(text, max(0.0, float(conf))))
        before = (self.text, self.confidence)
        self.text, self.confidence = self._consensus()
        return (self.text, self.confidence) != before

    @property
    def votes(self) -> int:
        """How many readings back the winning string, out of how many total."""
        return sum(1 for r in self.readings if r.text == self.text)

    def summary(self) -> str:
        """One line for a log: what won, and how contested it was."""
        tally: dict[str, int] = defaultdict(int)
        for r in self.readings:
            tally[r.text] += 1
        rival = " ".join(
            f"{t}x{n}" for t, n in sorted(tally.items(), key=lambda kv: -kv[1]) if t != self.text
        )
        return (
            f"{self.text} @{self.confidence:.2f} "
            f"({self.votes}/{len(self.readings)} reads"
            f"{'; also ' + rival if rival else ''})"
        )

    # ------------------------------------------------------------------
    def _consensus(self) -> tuple[str, float]:
        if not self.readings:
            return "", 0.0

        # 1. Group by shape, and let the shapes compete on total confidence
        #    rather than on count. One confident full read has to be able to
        #    outweigh three faint truncated ones, or a plate glimpsed clearly
        #    once and badly often would lose its leading letter to the crowd.
        groups: dict[tuple[int, int, int], list[_Reading]] = defaultdict(list)
        for r in self.readings:
            groups[plate_shape(r.text)].append(r)
        winner = max(
            groups.values(),
            # Total confidence, then how many readings, then the best single
            # one -- three keys so the outcome never depends on dict order.
            key=lambda g: (sum(r.conf for r in g), len(g), max(r.conf for r in g)),
        )

        # 2. Inside the winning shape every reading has the same length, so the
        #    columns are directly comparable. Each column is decided on its own.
        length = len(winner[0].text)
        chars: list[str] = []
        char_conf: list[float] = []
        for i in range(length):
            weight: dict[str, float] = defaultdict(float)
            best: dict[str, float] = defaultdict(float)
            for r in winner:
                ch = r.text[i]
                weight[ch] += r.conf
                best[ch] = max(best[ch], r.conf)
            # Total weight first, then the best single reading of that
            # character, then the character itself: all three so that a tie
            # resolves the same way every time it is recomputed.
            ch = max(weight, key=lambda c: (weight[c], best[c], c))
            chars.append(ch)
            char_conf.append(best[ch])
        consensus = "".join(chars)

        # 3. Confidence, kept on exactly the scale a single read reports so
        #    that ALPR_MIN_CONFIDENCE, the listing filter and rows written
        #    before any of this existed all still mean what they meant.
        exact = [r.conf for r in winner if r.text == consensus]
        if exact:
            # The consensus is a string some frame actually read: it can speak
            # for itself, and does, through its best sighting.
            return consensus, max(exact)
        # Assembled from columns that never appeared together in one frame.
        # Each column is priced at the best frame that read it, and the whole
        # is capped at the best single reading in the group -- a consensus is
        # allowed to be better than its parts, but not to claim a confidence no
        # frame ever supported.
        ceiling = max(r.conf for r in winner)
        return consensus, min(ceiling, sum(char_conf) / len(char_conf))
