import { ReactNode, useEffect, useRef, useState } from "react";
import {
  api,
  CLASS_LABELS,
  DETECTION_CLASSES,
  Plate,
  ReviewState,
  Sighting,
  reviewState,
} from "../api";

const STATUS_STYLES: Record<string, string> = {
  running: "bg-green-500/20 text-green-300 border-green-600",
  starting: "bg-yellow-500/20 text-yellow-300 border-yellow-600",
  stopped: "bg-slate-600/30 text-slate-300 border-slate-600",
  error: "bg-red-500/20 text-red-300 border-red-600",
};

export function StatusBadge({ status }: { status: string }) {
  const cls = STATUS_STYLES[status] ?? STATUS_STYLES.stopped;
  return (
    <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>{status}</span>
  );
}

export function ClassSelector({
  value,
  onChange,
}: {
  value: string[];
  onChange: (v: string[]) => void;
}) {
  const toggle = (c: string) => {
    onChange(value.includes(c) ? value.filter((x) => x !== c) : [...value, c]);
  };
  return (
    <div className="flex flex-wrap gap-2">
      {DETECTION_CLASSES.map((c) => (
        <button
          type="button"
          key={c}
          onClick={() => toggle(c)}
          className={`rounded-full border px-3 py-1 text-sm ${
            value.includes(c)
              ? "border-sky-500 bg-sky-500/20 text-sky-200"
              : "border-slate-700 text-slate-400"
          }`}
        >
          {CLASS_LABELS[c]}
        </button>
      ))}
    </div>
  );
}

// The shell both evidence modals are built from: the same overlay, the same
// escape-to-close, the same header strip over the same full-frame body. A plate
// read and a face sighting answer different questions but are looked at the
// same way -- one row, one moment, "show me the whole picture" -- so they open
// the same thing and only the header differs.
function EvidenceModal({
  open,
  onClose,
  frameUrl,
  missingFrameText,
  children,
  toolbar,
}: {
  open: boolean;
  onClose: () => void;
  /** null when nothing was stored for this row. */
  frameUrl: string | null;
  missingFrameText: string;
  children: ReactNode;
  /** An optional second strip under the header, for controls that act on the
   *  row rather than describe it. Kept off the header line because a row of
   *  facts and a row of inputs read badly interleaved. */
  toolbar?: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="max-h-full w-full max-w-5xl overflow-auto rounded-lg border border-slate-700 bg-slate-900"
      >
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-800 p-3">
          {children}
          <button
            onClick={onClose}
            className="ml-auto rounded bg-slate-800 px-3 py-1 text-sm hover:bg-slate-700"
          >
            Tutup
          </button>
        </div>
        {toolbar && (
          <div className="border-b border-slate-800 bg-slate-900/60 p-3">{toolbar}</div>
        )}
        {frameUrl ? (
          <img src={frameUrl} alt="frame penuh" className="block w-full" />
        ) : (
          <p className="p-8 text-center text-sm text-slate-500">{missingFrameText}</p>
        )}
      </div>
    </div>
  );
}

// Full-frame evidence for one plate read: the whole scene at the moment of the
// read with the vehicle boxed, so a person can confirm which vehicle a plate
// belongs to. The crop alone proves the characters but not what carried them.
const REVIEW_BADGES: Record<ReviewState, { label: string; className: string; title: string }> = {
  pending: {
    label: "Belum ditinjau",
    className: "border-slate-700 text-slate-400",
    title: "Belum ada orang yang memeriksa pembacaan ini",
  },
  correct: {
    label: "OCR benar",
    className: "border-green-700 bg-green-500/10 text-green-300",
    title: "Sudah diperiksa: OCR membacanya dengan benar",
  },
  wrong: {
    label: "Dikoreksi",
    className: "border-amber-600 bg-amber-500/10 text-amber-300",
    title: "Sudah diperiksa: OCR salah, nomor sebenarnya sudah dicatat",
  },
  illegible: {
    label: "Tidak terbaca",
    className: "border-slate-600 bg-slate-700/30 text-slate-400",
    title: "Sudah diperiksa: platnya memang tidak bisa dibaca dari gambar ini",
  },
};

/** One word on where a read stands with a reviewer. */
export function ReviewBadge({ plate }: { plate: Plate }) {
  const s = REVIEW_BADGES[reviewState(plate)];
  return (
    <span
      title={s.title}
      className={`whitespace-nowrap rounded-full border px-2 py-0.5 text-xs ${s.className}`}
    >
      {s.label}
    </span>
  );
}

// Full-frame evidence for one plate read, and the place a person corrects it.
//
// The two belong together. Correcting a plate means answering "what does this
// actually say?", and that question cannot be answered from the row in the
// table -- the crop there is 28 px tall and the frame is not shown at all. The
// modal is the only place both are on screen at once, so it is the only place
// where a correction is anything better than a guess.
//
// `onReviewed` is what makes this usable on more than one row at a time. There
// are over a thousand stored crops and labelling them is the whole point of
// having the feature, so saving moves to the next read rather than closing:
// type, Enter, type, Enter. `onStep` wires the arrows for skipping past ones
// that need no work.
export function PlateEvidenceModal({
  plate,
  onClose,
  onReviewed,
  onStep,
  position,
}: {
  plate: Plate | null;
  onClose: () => void;
  /** Called with the updated row after a save or an undo. */
  onReviewed?: (updated: Plate) => void;
  /** Move to the previous (-1) or next (+1) read in the current listing. */
  onStep?: (delta: number) => void;
  /** "3 / 15", for orientation while working through a page. */
  position?: string;
}) {
  return (
    <EvidenceModal
      open={plate !== null}
      onClose={onClose}
      frameUrl={plate?.has_frame ? api.plateFrameUrl(plate.id) : null}
      missingFrameText="Frame penuh tidak tersimpan untuk pembacaan ini."
      toolbar={
        plate && onReviewed ? (
          <PlateReviewBar
            plate={plate}
            onReviewed={onReviewed}
            onStep={onStep}
            position={position}
          />
        ) : undefined
      }
    >
      {plate && (
        <>
          {plate.has_image && (
            <img
              src={api.plateImageUrl(plate.id)}
              alt={plate.plate_text}
              /* Twice the height it used to be. This crop is what a reviewer
                 reads the plate off, and at h-10 they were reading the frame
                 instead and taking the crop on trust. */
              className="h-20 rounded border border-slate-700"
            />
          )}
          <span className="font-mono text-lg font-semibold tracking-wider">
            {plate.plate_text || <span className="font-sans text-sm text-slate-500">belum terbaca</span>}
          </span>
          <ReviewBadge plate={plate} />
          <span className="text-sm text-slate-400">{plate.vehicle_class}</span>
          <span className="font-mono text-sm text-slate-500">
            {(plate.confidence * 100).toFixed(0)}%
          </span>
          <span className="text-sm text-slate-500">
            {new Date(plate.timestamp).toLocaleString()}
          </span>
        </>
      )}
    </EvidenceModal>
  );
}

function PlateReviewBar({
  plate,
  onReviewed,
  onStep,
  position,
}: {
  plate: Plate;
  onReviewed: (updated: Plate) => void;
  onStep?: (delta: number) => void;
  position?: string;
}) {
  // Seeded from the best answer already on the row: a reviewer confirming a
  // correct read should press Enter, not retype eight characters. Re-seeded
  // per row, which is what the `plate.id` key does.
  const [text, setText] = useState(plate.corrected_text ?? plate.plate_text);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setText(plate.corrected_text ?? plate.plate_text);
    setError("");
    input.current?.focus();
    input.current?.select();
  }, [plate.id]);

  const save = async (value: string) => {
    setBusy(true);
    setError("");
    try {
      const updated = await api.reviewPlate(plate.id, value);
      onReviewed(updated);
      // Straight on to the next one. A reviewer working a page is in a rhythm,
      // and stopping to click "next" after every save is what breaks it.
      onStep?.(1);
    } catch (err: any) {
      setError(err?.message ?? "Gagal menyimpan");
    } finally {
      setBusy(false);
    }
  };

  const undo = async () => {
    setBusy(true);
    setError("");
    try {
      onReviewed(await api.unreviewPlate(plate.id));
    } catch (err: any) {
      setError(err?.message ?? "Gagal membatalkan");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <label htmlFor="review-plate" className="text-sm text-slate-400">
        Nomor sebenarnya
      </label>
      <input
        id="review-plate"
        ref={input}
        value={text}
        disabled={busy}
        onChange={(e) => setText(e.target.value.toUpperCase())}
        onKeyDown={(e) => {
          if (e.key === "Enter") save(text);
        }}
        placeholder="B 1234 XYZ"
        className="w-48 rounded border border-slate-700 bg-slate-800 px-3 py-2 font-mono text-base tracking-wider placeholder:font-sans placeholder:text-sm placeholder:tracking-normal placeholder:text-slate-500"
      />
      <button
        onClick={() => save(text)}
        disabled={busy || !text.trim()}
        title="Simpan dan lanjut ke pembacaan berikutnya (Enter)"
        className="rounded border border-sky-600 bg-sky-500/20 px-3 py-2 text-sm text-sky-200 hover:bg-sky-500/30 disabled:opacity-40"
      >
        Simpan &amp; lanjut
      </button>
      <button
        onClick={() => save("")}
        disabled={busy}
        title="Platnya memang tidak terbaca dari gambar ini -- dicatat begitu, bukan ditebak"
        className="rounded border border-slate-700 px-3 py-2 text-sm text-slate-400 hover:bg-slate-800"
      >
        Tidak terbaca
      </button>
      {plate.reviewed_at && (
        <button
          onClick={undo}
          disabled={busy}
          title="Batalkan tinjauan; baris kembali ke antrean"
          className="rounded border border-slate-700 px-3 py-2 text-sm text-slate-400 hover:bg-slate-800"
        >
          Batalkan tinjauan
        </button>
      )}

      <div className="ml-auto flex items-center gap-2 text-sm text-slate-500">
        {plate.reviewed_by && plate.reviewed_at && (
          <span title={new Date(plate.reviewed_at).toLocaleString()}>
            ditinjau {plate.reviewed_by}
          </span>
        )}
        {onStep && (
          <>
            {position && <span className="font-mono text-xs">{position}</span>}
            <button
              onClick={() => onStep(-1)}
              className="rounded border border-slate-700 px-2 py-2 hover:bg-slate-800"
              aria-label="Pembacaan sebelumnya"
            >
              ‹
            </button>
            <button
              onClick={() => onStep(1)}
              className="rounded border border-slate-700 px-2 py-2 hover:bg-slate-800"
              aria-label="Pembacaan berikutnya"
            >
              ›
            </button>
          </>
        )}
      </div>
      {error && <p className="w-full text-sm text-red-300">{error}</p>}
    </div>
  );
}

// The same thing for one face sighting: the whole scene with the face boxed.
// The crop says who was seen; only the frame says where they were, which way
// they were going, and who was with them.
export function SightingEvidenceModal({
  sighting,
  sourceName,
  onClose,
}: {
  sighting: Sighting | null;
  sourceName?: string;
  onClose: () => void;
}) {
  return (
    <EvidenceModal
      open={sighting !== null}
      onClose={onClose}
      frameUrl={sighting?.has_frame ? api.sightingFrameUrl(sighting.id) : null}
      missingFrameText="Frame penuh tidak tersimpan untuk kemunculan ini."
    >
      {sighting && (
        <>
          {sighting.has_image && (
            <img
              src={api.sightingImageUrl(sighting.id)}
              alt={sighting.name ?? "unknown"}
              className="h-10 w-10 rounded border border-slate-700 object-cover"
            />
          )}
          <span className="text-lg font-semibold">
            {sighting.name ?? <span className="text-slate-500">unknown</span>}
          </span>
          {sourceName && <span className="text-sm text-slate-400">{sourceName}</span>}
          {/* Similarity is only meaningful against a name it was matched to. */}
          {sighting.name && (
            <span className="font-mono text-sm text-slate-500">
              {(sighting.similarity * 100).toFixed(0)}%
            </span>
          )}
          <span className="text-sm text-slate-500">
            {new Date(sighting.timestamp).toLocaleString()}
          </span>
        </>
      )}
    </EvidenceModal>
  );
}
