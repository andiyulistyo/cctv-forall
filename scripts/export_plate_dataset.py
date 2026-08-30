#!/usr/bin/env python
"""Turn reviewed plate reads into a labelled dataset, and score OCR against it.

Reviewing a read in the dashboard records what the plate *actually* says next
to what OCR *said* it says, without disturbing either. That pairing is the only
ground truth this system has, and it is what two otherwise impossible jobs both
need:

* **Measuring.** Exact-match rate and character error rate for the current
  reader, on this camera, on these plates -- rather than on the benchmark
  somebody else's model was published against. Run with no arguments and that
  is all this prints; nothing is written.
* **Replacing.** A folder of plate crops with a labels file, which is what a
  purpose-built recogniser wants both to be fine-tuned on and to be compared
  against. ``--out`` writes one.

The split is by *vehicle*, not by row: several reads of one car share a plate,
and letting one land in train while another lands in eval leaks the answer and
flatters every score that follows.

The dashboard does all of this without a shell -- the **Akurasi OCR** page
shows the same numbers and downloads the same dataset as a zip. This script is
for the machine itself: cron, a build step, or writing straight into a training
directory rather than through a browser. Both sides share the arithmetic in
``app.plate_dataset``, so they cannot disagree about what the accuracy is.

    cd backend
    .venv/Scripts/python ../scripts/export_plate_dataset.py
    .venv/Scripts/python ../scripts/export_plate_dataset.py --out ../data/dataset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

from app.database import SessionLocal  # noqa: E402
from app.plate_dataset import (  # noqa: E402
    collect, review_progress, score, write_dataset,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, help="write crops and labels here (default: score only)")
    ap.add_argument(
        "--eval-share", type=float, default=0.2,
        help="fraction of plates held out for eval (default 0.2)",
    )
    args = ap.parse_args()

    db = SessionLocal()
    try:
        progress = review_progress(db)
        items, skipped = collect(db)
    finally:
        db.close()

    print("Reviewed plate reads:")
    print(f"  {progress['reviewed']} reviewed of {progress['total']} reads "
          f"({progress['pending']} still pending)")
    for reason, n in skipped.most_common():
        print(f"  skipped {n}: {reason}")

    if not items:
        print("\n  Nothing to score yet. Review some reads first: open the Plat Nomor")
        print("  page, filter to 'Belum ditinjau', and click a plate.")
        print("  Or use the Akurasi OCR page, which does all of this without a shell.")
        return 1

    m = score(items)
    print(f"\n  reads scored          {m.reads}")
    print(f"  exact match           {m.exact} ({m.exact_rate:.1%})")
    print(f"  nothing read at all   {m.blank} ({m.blank_rate:.1%})")
    print(f"  character error rate  {m.cer:.1%}  ({m.errors} edits over {m.chars} chars)")
    if m.confusions:
        print("  most confused         " + "  ".join(f"{p} x{n}" for p, n in m.confusions))

    if args.out:
        counts = write_dataset(items, args.out, args.eval_share)
        print(f"\n  wrote {counts['train']} train / {counts['eval']} eval to {args.out}")
        print("  labels.jsonl (full record) and labels.csv (path,label)")
    else:
        print("\n  (scored only -- pass --out DIR to write the dataset)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
