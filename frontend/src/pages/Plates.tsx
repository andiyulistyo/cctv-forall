import { useEffect, useState } from "react";
import { api, Plate, Source } from "../api";
import { PlateEvidenceModal } from "../components/common";

export default function Plates() {
  const [plates, setPlates] = useState<Plate[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [sourceId, setSourceId] = useState<number | undefined>(undefined);
  const [evidence, setEvidence] = useState<Plate | null>(null);

  const load = async () => {
    setPlates(await api.listPlates(sourceId, 200));
  };

  useEffect(() => {
    api.listSources().then((r) => setSources(r.sources));
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [sourceId]);

  const nameOf = (id: number) => sources.find((s) => s.id === id)?.name ?? `#${id}`;

  return (
    <div>
      <PlateEvidenceModal plate={evidence} onClose={() => setEvidence(null)} />
      <div className="mb-5 flex items-center gap-3">
        <h1 className="text-2xl font-semibold">Plat Nomor</h1>
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
              <th className="p-3">Plat (crop)</th>
              <th className="p-3">Kendaraan</th>
              <th className="p-3">Nomor</th>
              <th className="p-3">Kelas</th>
              <th className="p-3">Source</th>
              <th className="p-3">Confidence</th>
              <th className="p-3">Waktu</th>
            </tr>
          </thead>
          <tbody>
            {plates.map((p) => (
              <tr key={p.id} className="border-t border-slate-800">
                <td className="p-3">
                  {p.has_image ? (
                    <img
                      src={api.plateImageUrl(p.id)}
                      alt={p.plate_text}
                      className="h-10 w-28 rounded border border-slate-700 object-cover"
                    />
                  ) : (
                    <span className="text-slate-600">—</span>
                  )}
                </td>
                <td className="p-3">
                  {p.has_frame ? (
                    <img
                      src={api.plateFrameUrl(p.id)}
                      alt="kendaraan"
                      loading="lazy"
                      onClick={() => setEvidence(p)}
                      title="Lihat frame penuh"
                      className="h-16 w-28 cursor-zoom-in rounded border border-slate-700 object-cover hover:border-sky-500"
                    />
                  ) : (
                    <span className="text-slate-600">—</span>
                  )}
                </td>
                <td className="p-3 font-mono text-base font-semibold tracking-wider">
                  {p.plate_text}
                </td>
                <td className="p-3 text-slate-300">{p.vehicle_class}</td>
                <td className="p-3 text-slate-300">{nameOf(p.source_id)}</td>
                <td className="p-3 font-mono text-slate-400">{(p.confidence * 100).toFixed(0)}%</td>
                <td className="p-3 text-slate-400">{new Date(p.timestamp).toLocaleString()}</td>
              </tr>
            ))}
            {plates.length === 0 && (
              <tr>
                <td colSpan={7} className="p-8 text-center text-slate-500">
                  Belum ada data plat nomor.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
