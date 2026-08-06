import { useState } from "react";
import { api, SourceType } from "../api";
import { ClassSelector } from "./common";

const SOURCE_TYPES: { value: SourceType; label: string; hint: string }[] = [
  { value: "youtube", label: "YouTube", hint: "https://www.youtube.com/watch?v=..." },
  { value: "rtsp", label: "RTSP / CCTV", hint: "rtsp://user:pass@ip:554/stream" },
  { value: "rtmp", label: "RTMP", hint: "rtmp://server/live/key" },
  { value: "hls", label: "HLS", hint: "https://.../index.m3u8" },
  { value: "http", label: "HTTP / MJPEG", hint: "http://.../video.mjpg" },
  { value: "file", label: "File", hint: "/data/video.mp4" },
];

export default function AddSourceModal({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: () => void;
}) {
  const [name, setName] = useState("");
  const [type, setType] = useState<SourceType>("youtube");
  const [url, setUrl] = useState("");
  const [classes, setClasses] = useState<string[]>(["car", "truck", "motorcycle"]);
  const [alpr, setAlpr] = useState(true);
  const [face, setFace] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const hint = SOURCE_TYPES.find((t) => t.value === type)?.hint;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.createSource({
        name,
        type,
        url,
        enabled_classes: classes,
        alpr_enabled: alpr,
        face_enabled: face,
      });
      onCreated();
      onClose();
    } catch (err: any) {
      setError(err.message || "Gagal menambah source");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <form
        onSubmit={submit}
        className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-xl border border-slate-800 bg-slate-900 p-6"
      >
        <h2 className="mb-4 text-lg font-semibold">Tambah Source</h2>

        <label className="mb-1 block text-sm text-slate-300">Nama</label>
        <input
          className="mb-3 w-full rounded border border-slate-700 bg-slate-800 px-3 py-2"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Kamera Depan"
          required
        />

        <label className="mb-1 block text-sm text-slate-300">Tipe</label>
        <select
          className="mb-3 w-full rounded border border-slate-700 bg-slate-800 px-3 py-2"
          value={type}
          onChange={(e) => setType(e.target.value as SourceType)}
        >
          {SOURCE_TYPES.map((t) => (
            <option key={t.value} value={t.value}>
              {t.label}
            </option>
          ))}
        </select>

        <label className="mb-1 block text-sm text-slate-300">URL / Path</label>
        <input
          className="mb-1 w-full rounded border border-slate-700 bg-slate-800 px-3 py-2"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder={hint}
          required
        />
        <p className="mb-3 text-xs text-slate-500">{hint}</p>

        <label className="mb-1 block text-sm text-slate-300">Objek yang dideteksi</label>
        <div className="mb-3">
          <ClassSelector value={classes} onChange={setClasses} />
        </div>

        <label className="mb-2 flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={alpr} onChange={(e) => setAlpr(e.target.checked)} />
          Baca plat nomor (ANPR) untuk kendaraan
        </label>

        <label className="mb-4 flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={face} onChange={(e) => setFace(e.target.checked)} />
          Pengenalan wajah (face recognition)
        </label>

        {error && <div className="mb-3 text-sm text-red-400">{error}</div>}

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded bg-slate-800 px-4 py-2 hover:bg-slate-700"
          >
            Batal
          </button>
          <button
            disabled={busy || classes.length === 0}
            className="rounded bg-sky-600 px-4 py-2 font-medium hover:bg-sky-500 disabled:opacity-50"
          >
            {busy ? "Menyimpan..." : "Simpan"}
          </button>
        </div>
      </form>
    </div>
  );
}
