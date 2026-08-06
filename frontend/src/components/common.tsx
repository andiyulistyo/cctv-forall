import { DETECTION_CLASSES } from "../api";

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
