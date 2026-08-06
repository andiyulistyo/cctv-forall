import { useEffect, useRef, useState } from "react";
import { api, EnrolledFace, Sighting, Source } from "../api";

export default function Faces() {
  const [faces, setFaces] = useState<EnrolledFace[]>([]);
  const [sightings, setSightings] = useState<Sighting[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [sourceId, setSourceId] = useState<number | undefined>(undefined);

  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const loadFaces = async () => setFaces(await api.listFaces());
  const loadSightings = async () => setSightings(await api.listSightings(sourceId, 200));

  useEffect(() => {
    loadFaces();
    api.listSources().then((r) => setSources(r.sources));
  }, []);

  useEffect(() => {
    loadSightings();
    const t = setInterval(loadSightings, 5000);
    return () => clearInterval(t);
  }, [sourceId]);

  const enroll = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !name) return;
    setBusy(true);
    setError("");
    try {
      await api.enrollFace(name, file);
      setName("");
      setFile(null);
      if (fileRef.current) fileRef.current.value = "";
      loadFaces();
    } catch (err: any) {
      setError(err.message || "Enrol gagal");
    } finally {
      setBusy(false);
    }
  };

  const remove = async (f: EnrolledFace) => {
    if (!confirm(`Hapus wajah terdaftar "${f.name}"?`)) return;
    await api.deleteFace(f.id);
    loadFaces();
  };

  const nameOf = (id: number) => sources.find((s) => s.id === id)?.name ?? `#${id}`;

  return (
    <div>
      <h1 className="mb-5 text-2xl font-semibold">Pengenalan Wajah</h1>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Enrol + enrolled list */}
        <div className="space-y-6">
          <form onSubmit={enroll} className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="mb-3 font-medium">Daftarkan Orang</h2>
            <label className="mb-1 block text-sm text-slate-300">Nama</label>
            <input
              className="mb-3 w-full rounded border border-slate-700 bg-slate-800 px-3 py-2"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Budi Santoso"
              required
            />
            <label className="mb-1 block text-sm text-slate-300">Foto wajah (frontal jelas)</label>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="mb-3 w-full text-sm text-slate-400"
              required
            />
            {error && <div className="mb-3 text-sm text-red-400">{error}</div>}
            <button
              disabled={busy || !file || !name}
              className="w-full rounded bg-sky-600 px-3 py-2 font-medium hover:bg-sky-500 disabled:opacity-50"
            >
              {busy ? "Memproses..." : "Daftarkan"}
            </button>
          </form>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <h2 className="mb-3 font-medium">Terdaftar ({faces.length})</h2>
            {faces.length === 0 ? (
              <p className="text-sm text-slate-500">Belum ada wajah terdaftar.</p>
            ) : (
              <ul className="space-y-2">
                {faces.map((f) => (
                  <li key={f.id} className="flex items-center gap-3">
                    {f.has_image && (
                      <img
                        src={api.faceImageUrl(f.id)}
                        alt={f.name}
                        className="h-12 w-12 rounded-full border border-slate-700 object-cover"
                      />
                    )}
                    <span className="font-medium">{f.name}</span>
                    <button
                      onClick={() => remove(f)}
                      className="ml-auto rounded bg-red-900/50 px-2 py-1 text-xs text-red-300 hover:bg-red-900"
                    >
                      Hapus
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Sightings */}
        <div className="lg:col-span-2">
          <div className="mb-3 flex items-center gap-3">
            <h2 className="font-medium">Kemunculan Terdeteksi</h2>
            <select
              className="ml-auto rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
              value={sourceId ?? ""}
              onChange={(e) => setSourceId(e.target.value ? Number(e.target.value) : undefined)}
            >
              <option value="">Semua source</option>
              {sources.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
          <div className="overflow-x-auto rounded-lg border border-slate-800">
            <table className="w-full text-sm">
              <thead className="bg-slate-900 text-left text-slate-400">
                <tr>
                  <th className="p-3">Wajah</th>
                  <th className="p-3">Nama</th>
                  <th className="p-3">Source</th>
                  <th className="p-3">Similarity</th>
                  <th className="p-3">Waktu</th>
                </tr>
              </thead>
              <tbody>
                {sightings.map((s) => (
                  <tr key={s.id} className="border-t border-slate-800">
                    <td className="p-3">
                      {s.has_image ? (
                        <img
                          src={api.sightingImageUrl(s.id)}
                          alt={s.name ?? "unknown"}
                          className="h-12 w-12 rounded border border-slate-700 object-cover"
                        />
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                    <td className="p-3 font-medium">
                      {s.name ?? <span className="text-slate-500">unknown</span>}
                    </td>
                    <td className="p-3 text-slate-300">{nameOf(s.source_id)}</td>
                    <td className="p-3 font-mono text-slate-400">
                      {s.name ? (s.similarity * 100).toFixed(0) + "%" : "—"}
                    </td>
                    <td className="p-3 text-slate-400">{new Date(s.timestamp).toLocaleString()}</td>
                  </tr>
                ))}
                {sightings.length === 0 && (
                  <tr>
                    <td colSpan={5} className="p-8 text-center text-slate-500">
                      Belum ada kemunculan wajah terdeteksi.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
