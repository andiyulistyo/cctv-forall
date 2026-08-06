import { useEffect, useRef, useState } from "react";
import { api, Source } from "../api";

// Lets the user draw a single counting line over a snapshot of the source by
// clicking two points. Coordinates are stored normalized (0..1) so they stay
// valid regardless of the stream resolution.
export default function LineDrawCanvas({
  source,
  onSaved,
}: {
  source: Source;
  onSaved: () => void;
}) {
  const [imgUrl, setImgUrl] = useState<string>("");
  const [loadErr, setLoadErr] = useState("");
  const [points, setPoints] = useState<number[][]>(
    source.line ? [source.line.a, source.line.b] : []
  );
  const [inLabel, setInLabel] = useState(source.direction_labels?.in ?? "masuk");
  const [outLabel, setOutLabel] = useState(source.direction_labels?.out ?? "keluar");
  const [saving, setSaving] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let revoked = "";
    api
      .snapshotBlobUrl(source.id)
      .then((u) => {
        revoked = u;
        setImgUrl(u);
      })
      .catch(() => setLoadErr("Tidak bisa mengambil snapshot. Pastikan source dapat diakses."));
    return () => {
      if (revoked) URL.revokeObjectURL(revoked);
    };
  }, [source.id]);

  const onClick = (e: React.MouseEvent) => {
    const box = boxRef.current;
    if (!box) return;
    const rect = box.getBoundingClientRect();
    const nx = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const ny = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
    setPoints((prev) => (prev.length >= 2 ? [[nx, ny]] : [...prev, [nx, ny]]));
  };

  const save = async () => {
    if (points.length !== 2) return;
    setSaving(true);
    try {
      await api.setLine(source.id, points[0], points[1], { in: inLabel, out: outLabel });
      onSaved();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <p className="mb-2 text-sm text-slate-400">
        Klik dua titik pada gambar untuk membuat garis hitung. Objek yang melintasi garis akan
        dihitung. Arah <b>{inLabel}</b> / <b>{outLabel}</b> ditentukan sisi garis.
      </p>
      <div
        ref={boxRef}
        onClick={onClick}
        className="relative w-full cursor-crosshair select-none overflow-hidden rounded-lg border border-slate-700 bg-black"
      >
        {imgUrl ? (
          <img src={imgUrl} alt="snapshot" className="block w-full" draggable={false} />
        ) : (
          <div className="flex aspect-video items-center justify-center text-slate-500">
            {loadErr || "Memuat snapshot..."}
          </div>
        )}
        <svg className="pointer-events-none absolute inset-0 h-full w-full">
          {points.length === 2 && (
            <line
              x1={`${points[0][0] * 100}%`}
              y1={`${points[0][1] * 100}%`}
              x2={`${points[1][0] * 100}%`}
              y2={`${points[1][1] * 100}%`}
              stroke="#22d3ee"
              strokeWidth={3}
            />
          )}
          {points.map((p, i) => (
            <circle key={i} cx={`${p[0] * 100}%`} cy={`${p[1] * 100}%`} r={6} fill="#22d3ee" />
          ))}
        </svg>
      </div>

      <div className="mt-3 flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-xs text-slate-400">Label arah 1 (sisi A→B)</label>
          <input
            className="w-28 rounded border border-slate-700 bg-slate-800 px-2 py-1 text-sm"
            value={inLabel}
            onChange={(e) => setInLabel(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-xs text-slate-400">Label arah 2</label>
          <input
            className="w-28 rounded border border-slate-700 bg-slate-800 px-2 py-1 text-sm"
            value={outLabel}
            onChange={(e) => setOutLabel(e.target.value)}
          />
        </div>
        <button
          onClick={() => setPoints([])}
          className="rounded bg-slate-800 px-3 py-1.5 text-sm hover:bg-slate-700"
        >
          Reset
        </button>
        <button
          onClick={save}
          disabled={points.length !== 2 || saving}
          className="rounded bg-sky-600 px-4 py-1.5 text-sm font-medium hover:bg-sky-500 disabled:opacity-50"
        >
          {saving ? "Menyimpan..." : "Simpan Garis"}
        </button>
      </div>
    </div>
  );
}
