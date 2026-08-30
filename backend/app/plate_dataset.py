"""Reviewed plate reads, turned into a score and into a training set.

Reviewing a read in the dashboard records what the plate *actually* says beside
what OCR said it says, without disturbing either (see ``models.PlateRead``).
That pairing is the only ground truth this system has, and two otherwise
impossible jobs both run on it:

* **Measuring.** Exact-match rate and character error rate for the reader you
  are actually running, on your cameras, on Indonesian plates -- rather than on
  the benchmark somebody else's model was published against.
* **Replacing.** A folder of plate crops with a labels file, which is what a
  purpose-built recogniser wants both to be fine-tuned on and to be judged
  against.

This module holds the arithmetic for both. It is deliberately separate from
the API that serves it and the script that prints it, because those two must
never be able to disagree about what the accuracy *is* -- a dashboard and a
CLI quoting different numbers from the same rows is worse than having neither.

No OpenCV, no model, no HTTP: rows in, numbers and files out.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import func, select

from .config import settings
from .models import PlateRead


@dataclass
class DatasetItem:
    """One reviewed read: a picture, what OCR made of it, and what it is."""

    id: int
    source_id: int
    label: str          # what a person says the plate is
    predicted: str      # what OCR said -- never overwritten by the review
    confidence: float
    reviewed_by: str | None
    timestamp: str
    crop: Path

    def as_record(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "label": self.label,
            "predicted": self.predicted,
            "confidence": round(self.confidence, 4),
            "reviewed_by": self.reviewed_by,
            "timestamp": self.timestamp,
        }


@dataclass
class Metrics:
    """How the current reader does against the labels."""

    reads: int = 0
    exact: int = 0
    blank: int = 0          # OCR returned nothing at all
    chars: int = 0          # characters in the labels
    errors: int = 0         # edits needed to turn predictions into labels
    confusions: list[tuple[str, int]] = field(default_factory=list)

    @property
    def exact_rate(self) -> float:
        return self.exact / self.reads if self.reads else 0.0

    @property
    def blank_rate(self) -> float:
        return self.blank / self.reads if self.reads else 0.0

    @property
    def cer(self) -> float:
        """Character error rate: edits per character of ground truth."""
        return self.errors / self.chars if self.chars else 0.0

    def as_dict(self) -> dict:
        return {
            "reads": self.reads,
            "exact": self.exact,
            "exact_rate": round(self.exact_rate, 4),
            "blank": self.blank,
            "blank_rate": round(self.blank_rate, 4),
            "chars": self.chars,
            "errors": self.errors,
            "cer": round(self.cer, 4),
            "confusions": [{"pair": p, "count": n} for p, n in self.confusions],
        }


def levenshtein(a: str, b: str) -> int:
    """Edit distance, for character error rate.

    Exact match alone is a harsh and fairly uninformative score: it cannot tell
    a reader that misses one character on every plate from one that returns
    noise, and those two want completely different responses. Two rows of ints
    is all the space this needs at plate length.
    """
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def split_for(plate: str, eval_share: float) -> str:
    """Which split a plate belongs to -- decided by the plate, not the row.

    Hashing the label rather than shuffling rows does two things at once: every
    read of one vehicle lands in the same split (several reads of one car share
    a plate, and letting them straddle the boundary leaks the answer into the
    eval set), and the split is stable across runs, so a score from today is
    comparable with one from a fortnight ago instead of being a fresh sample of
    the same data.
    """
    digest = hashlib.sha1(plate.encode("utf-8")).digest()
    return "eval" if (digest[0] / 256.0) < eval_share else "train"


def review_progress(db) -> dict:
    """How much of the traffic has been looked at, and how it came out.

    Counted in SQL rather than by loading the rows: this runs on every load of
    the accuracy page, against a table that grows all day.
    """
    total = db.scalar(select(func.count()).select_from(PlateRead)) or 0
    reviewed = db.scalar(
        select(func.count()).select_from(PlateRead).where(PlateRead.reviewed_at.is_not(None))
    ) or 0
    illegible = db.scalar(
        select(func.count()).select_from(PlateRead).where(
            PlateRead.reviewed_at.is_not(None), PlateRead.corrected_text == ""
        )
    ) or 0
    correct = db.scalar(
        select(func.count()).select_from(PlateRead).where(
            PlateRead.reviewed_at.is_not(None),
            PlateRead.corrected_text != "",
            PlateRead.corrected_text == PlateRead.plate_text,
        )
    ) or 0
    return {
        "total": total,
        "reviewed": reviewed,
        "pending": total - reviewed,
        "correct": correct,
        # Legible, reviewed, and OCR disagreed -- the error set.
        "wrong": reviewed - illegible - correct,
        "illegible": illegible,
    }


def collect(db) -> tuple[list[DatasetItem], Counter]:
    """Every reviewed, legible read whose crop is still on disk.

    Returns the items and a tally of what was left out and why -- the second
    half matters, because "412 reads scored" means something different when
    300 more were dropped for a missing file.
    """
    rows = db.scalars(
        select(PlateRead)
        .where(PlateRead.reviewed_at.is_not(None))
        .order_by(PlateRead.id)
    ).all()

    items: list[DatasetItem] = []
    skipped: Counter = Counter()
    for r in rows:
        if not r.corrected_text:
            # Reviewed as illegible. Excluded on purpose: a crop no human can
            # read is not a training example, and scoring the reader against a
            # plate that is not visible measures nothing. The click was still
            # worth making -- this is the difference between "excluded, known
            # unreadable" and "never looked at".
            skipped["reviewed as illegible"] += 1
            continue
        if not r.image_path:
            skipped["no crop recorded"] += 1
            continue
        crop = settings.data_dir / r.image_path
        if not crop.exists():
            skipped["crop missing from disk"] += 1
            continue
        items.append(DatasetItem(
            id=r.id,
            source_id=r.source_id,
            label=r.corrected_text,
            predicted=r.plate_text,
            confidence=r.confidence,
            reviewed_by=r.reviewed_by,
            timestamp=r.timestamp.isoformat(),
            crop=crop,
        ))
    return items, skipped


def score(items: list[DatasetItem]) -> Metrics:
    """Exact match, character error rate, and what the reader confuses."""
    m = Metrics(reads=len(items))
    if not items:
        return m

    confusions: Counter = Counter()
    for i in items:
        if i.predicted == i.label:
            m.exact += 1
        if not i.predicted:
            m.blank += 1
        m.chars += len(i.label)
        m.errors += levenshtein(i.predicted, i.label)
        # Only same-length misreads: there the pairing between a predicted
        # character and the true one is unambiguous, and a wrong pair invented
        # from a length mismatch would poison the whole table. This is the list
        # a repair table (see alpr._CONFUSIONS) should be built from -- measured
        # rather than guessed at.
        if i.predicted and i.predicted != i.label and len(i.predicted) == len(i.label):
            confusions.update(f"{a}->{b}" for a, b in zip(i.predicted, i.label) if a != b)
    m.confusions = confusions.most_common(12)
    return m


def score_by_source(items: list[DatasetItem]) -> dict[int, Metrics]:
    """The same numbers per camera.

    Worth its own function because it is the most actionable thing here: a
    reader is not uniformly bad, it is bad on the camera pointed too high or
    aimed into the afternoon sun, and that is a problem you fix with a ladder
    rather than with a model.
    """
    grouped: dict[int, list[DatasetItem]] = {}
    for i in items:
        grouped.setdefault(i.source_id, []).append(i)
    return {sid: score(rows) for sid, rows in grouped.items()}


def _manifest(items: list[DatasetItem], eval_share: float) -> list[dict]:
    """One record per item, with its split and the path it will be written to."""
    out = []
    for item in items:
        split = split_for(item.label, eval_share)
        # Named by row id, not by plate: two reads of one plate would otherwise
        # collide and the second would silently replace the first, quietly
        # shrinking the dataset every time a plate is seen twice.
        name = f"{item.id}_{item.label}.jpg"
        out.append({**item.as_record(), "split": split, "path": f"{split}/{name}"})
    return out


def _labels_csv(manifest: list[dict]) -> str:
    """The plain two-column form most training scripts read."""
    import io

    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(["path", "label"])
    for row in manifest:
        w.writerow([row["path"], row["label"]])
    return buf.getvalue()


def _labels_jsonl(manifest: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest)


def write_dataset(items: list[DatasetItem], out: Path, eval_share: float) -> Counter:
    """Write crops and labels into a directory. Returns per-split counts."""
    manifest = _manifest(items, eval_share)
    counts: Counter = Counter()
    for item, row in zip(items, manifest):
        counts[row["split"]] += 1
        dest = out / row["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.crop, dest)
    (out / "labels.jsonl").write_text(_labels_jsonl(manifest), encoding="utf-8")
    (out / "labels.csv").write_text(_labels_csv(manifest), encoding="utf-8")
    return counts


def write_archive(items: list[DatasetItem], dest: Path, eval_share: float) -> Counter:
    """The same dataset as one zip, for downloading out of the dashboard.

    Written to a file rather than built in memory: the crops average ~35 KB and
    a fully reviewed camera-year is a few hundred megabytes, which is not a
    thing to hold in the web process while a browser reads it slowly.
    """
    manifest = _manifest(items, eval_share)
    counts: Counter = Counter()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for item, row in zip(items, manifest):
            counts[row["split"]] += 1
            # Stored, not deflated: these are JPEGs. Recompressing them costs
            # CPU for a percent or so, on the largest part of the archive.
            zf.write(item.crop, row["path"], compress_type=zipfile.ZIP_STORED)
        zf.writestr("labels.jsonl", _labels_jsonl(manifest))
        zf.writestr("labels.csv", _labels_csv(manifest))
        zf.writestr("README.txt", _archive_readme(counts, eval_share))
    return counts


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _archive_readme(counts: Counter, eval_share: float) -> str:
    """What is in the zip, for whoever opens it a month from now."""
    return (
        "Plate dataset exported from the CCTV detection dashboard.\n"
        "\n"
        f"  train/   {_plural(counts['train'], 'crop')}\n"
        f"  eval/    {_plural(counts['eval'], 'crop')}   (held out: {eval_share:.0%} of plates)\n"
        "  labels.csv     path,label -- what most training scripts read\n"
        "  labels.jsonl   the same rows plus what OCR predicted, its confidence,\n"
        "                 the source id, the timestamp and who reviewed it\n"
        "\n"
        "Every label is a human's reading of the crop next to it. Reads that a\n"
        "reviewer marked unreadable are NOT here: a crop nobody can read is not\n"
        "a training example.\n"
        "\n"
        "The split is by plate, not by row. Several reads of one vehicle share a\n"
        "plate, and letting them fall on both sides of the split would put the\n"
        "answer in the eval set and flatter every score measured against it.\n"
    )
