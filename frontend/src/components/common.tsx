import { useEffect } from "react";
import { api, DETECTION_CLASSES, Plate } from "../api";

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
  const LABELS: Record<string, string> = {
    person: "Orang",
    car: "Mobil",
    motorcycle: "Motor",
    truck: "Truk",
    bus: "Bus",
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
          {LABELS[c]}
        </button>
      ))}
    </div>
  );
}

// Full-frame evidence for one plate read: the whole scene at the moment of the
// read with the vehicle boxed, so a person can confirm which vehicle a plate
// belongs to. The crop alone proves the characters but not what carried them.
export function PlateEvidenceModal({
  plate,
  onClose,
}: {
  plate: Plate | null;
  onClose: () => void;
}) {
  useEffect(() => {
    if (!plate) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [plate, onClose]);

  if (!plate) return null;
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
          {plate.has_image && (
            <img
              src={api.plateImageUrl(plate.id)}
              alt={plate.plate_text}
              className="h-10 rounded border border-slate-700"
            />
          )}
          <span className="font-mono text-lg font-semibold tracking-wider">
            {plate.plate_text}
          </span>
          <span className="text-sm text-slate-400">{plate.vehicle_class}</span>
          <span className="font-mono text-sm text-slate-500">
            {(plate.confidence * 100).toFixed(0)}%
          </span>
          <span className="text-sm text-slate-500">
            {new Date(plate.timestamp).toLocaleString()}
          </span>
          <button
            onClick={onClose}
            className="ml-auto rounded bg-slate-800 px-3 py-1 text-sm hover:bg-slate-700"
          >
            Tutup
          </button>
        </div>
        {plate.has_frame ? (
          <img
            src={api.plateFrameUrl(plate.id)}
            alt="frame penuh"
            className="block w-full"
          />
        ) : (
          <p className="p-8 text-center text-sm text-slate-500">
            Frame penuh tidak tersimpan untuk pembacaan ini.
          </p>
        )}
      </div>
    </div>
  );
}
