import { useEffect, useRef, useState } from "react";
import { api, Source } from "../api";

type Corner = { x: number; y: number };

// Lets the user drag a rectangle over a snapshot to mark where plates should be
// read. Coordinates are stored normalized (0..1) so the zone survives a change
// of stream resolution, and the two corners are kept in drag order — the worker
// sorts them, so dragging from any corner works.
export default function ZoneDrawCanvas({
  source,
  onSaved,
}: {
  source: Source;
  onSaved: () => void;
}) {
  const [imgUrl, setImgUrl] = useState("");
  const [loadErr, setLoadErr] = useState("");
  const [start, setStart] = useState<Corner | null>(null);
  const [end, setEnd] = useState<Corner | null>(null);
  const [dragging, setDragging] = useState(false);
  const [saving, setSaving] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (source.alpr_zone) {
      setStart({ x: source.alpr_zone.a[0], y: source.alpr_zone.a[1] });
      setEnd({ x: source.alpr_zone.b[0], y: source.alpr_zone.b[1] });
    }
  }, [source.id]);

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

  const pointAt = (e: React.MouseEvent): Corner | null => {
    const box = boxRef.current;
    if (!box) return null;
    const rect = box.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)),
    };
  };

  const onMouseDown = (e: React.MouseEvent) => {
    const p = pointAt(e);
    if (!p) return;
    setStart(p);
    setEnd(p);
    setDragging(true);
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!dragging) return;
    const p = pointAt(e);
    if (p) setEnd(p);
  };
  // Also ends on mouse-leave: releasing the button outside the image would
  // otherwise leave the drag stuck on and every later hover would resize it.
  const stopDrag = () => setDragging(false);

  // Normalized rect for rendering, independent of which way the drag went.
  const rect =
    start && end
      ? {
          left: Math.min(start.x, end.x),
          top: Math.min(start.y, end.y),
          width: Math.abs(end.x - start.x),
          height: Math.abs(end.y - start.y),
        }
      : null;
  const tooSmall = !rect || rect.width < 0.02 || rect.height < 0.02;

  const save = async () => {
    if (!start || !end || tooSmall) return;
    setSaving(true);
    try {
      await api.setAlprZone(source.id, { a: [start.x, start.y], b: [end.x, end.y] });
      onSaved();
    } finally {
      setSaving(false);
    }
  };

  const clear = async () => {
    setSaving(true);
    try {
      await api.setAlprZone(source.id, null);
      setStart(null);
      setEnd(null);
      onSaved();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <p className="mb-2 text-sm text-slate-400">
        Tarik (drag) sebuah kotak pada bagian gambar tempat plat nomor <b>paling jelas terbaca</b> —
        biasanya area terdekat dengan kamera, tepat sebelum atau di sekitar garis hitung. Program
        hanya akan mencoba membaca plat saat titik roda kendaraan masuk kotak ini.
      </p>
      <div
        ref={boxRef}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={stopDrag}
        onMouseLeave={stopDrag}
        className="relative w-full cursor-crosshair select-none overflow-hidden rounded-lg border border-slate-700 bg-black"
      >
        {imgUrl ? (
          <img src={imgUrl} alt="snapshot" className="block w-full" draggable={false} />
        ) : (
          <div className="flex aspect-video items-center justify-center text-slate-500">
            {loadErr || "Memuat snapshot..."}
          </div>
        )}

        {/* Existing counting line, for context: the zone is usually placed around it. */}
        <svg className="pointer-events-none absolute inset-0 h-full w-full">
          {source.line && (
            <line
              x1={`${source.line.a[0] * 100}%`}
              y1={`${source.line.a[1] * 100}%`}
              x2={`${source.line.b[0] * 100}%`}
              y2={`${source.line.b[1] * 100}%`}
              stroke="#facc15"
              strokeWidth={2}
              strokeDasharray="6 4"
            />
          )}
        </svg>

        {rect && (
          <div
            className="pointer-events-none absolute border-2 border-cyan-400 bg-cyan-400/10"
            style={{
              left: `${rect.left * 100}%`,
              top: `${rect.top * 100}%`,
              width: `${rect.width * 100}%`,
              height: `${rect.height * 100}%`,
            }}
          >
            <span className="absolute left-1 top-1 rounded bg-cyan-400 px-1 text-[10px] font-semibold text-slate-900">
              ANPR
            </span>
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <span className="text-xs text-slate-500">
          {source.alpr_zone
            ? "Zona aktif — plat hanya dibaca di dalam kotak."
            : "Belum ada zona — plat dibaca di seluruh frame."}
        </span>
        <button
          onClick={clear}
          disabled={saving}
          className="ml-auto rounded bg-slate-800 px-3 py-1.5 text-sm hover:bg-slate-700 disabled:opacity-50"
        >
          Hapus Zona
        </button>
        <button
          onClick={save}
          disabled={tooSmall || saving}
          className="rounded bg-sky-600 px-4 py-1.5 text-sm font-medium hover:bg-sky-500 disabled:opacity-50"
        >
          {saving ? "Menyimpan..." : "Simpan Zona"}
        </button>
      </div>
      <p className="mt-2 text-xs text-slate-500">
        Menyimpan akan me-restart source ini agar zona langsung berlaku.
      </p>
    </div>
  );
}
