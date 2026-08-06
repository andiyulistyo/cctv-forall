import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, CountsResponse, Plate, Sighting, Source } from "../api";
import { StatusBadge } from "../components/common";
import LineDrawCanvas from "../components/LineDrawCanvas";

type Tab = "live" | "line";

export default function SourceDetail() {
  const { id } = useParams();
  const sourceId = Number(id);
  const [source, setSource] = useState<Source | null>(null);
  const [counts, setCounts] = useState<CountsResponse | null>(null);
  const [plates, setPlates] = useState<Plate[]>([]);
  const [sightings, setSightings] = useState<Sighting[]>([]);
  const [tab, setTab] = useState<Tab>("live");

  const loadSource = async () => setSource(await api.getSource(sourceId));

  useEffect(() => {
    loadSource();
  }, [sourceId]);

  useEffect(() => {
    const load = async () => {
      try {
        setCounts(await api.getCounts(sourceId));
        setPlates(await api.listPlates(sourceId, 20));
        setSightings(await api.listSightings(sourceId, 20));
        setSource(await api.getSource(sourceId));
      } catch {
        /* ignore transient */
      }
    };
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [sourceId]);

  if (!source) return <div className="text-slate-400">Memuat...</div>;

  const running = source.status === "running" || source.status === "starting";
  const toggle = async () => {
    if (running) await api.stopSource(sourceId);
    else await api.startSource(sourceId);
    loadSource();
  };

  // Merge live counts (from running worker) with DB totals as fallback.
  const live = counts?.live ?? {};
  const dbByClass: Record<string, { in: number; out: number }> = {};
  for (const b of counts?.buckets ?? []) {
    dbByClass[b.class_name] = dbByClass[b.class_name] ?? { in: 0, out: 0 };
    (dbByClass[b.class_name] as any)[b.direction] = b.count;
  }
  const classes = Array.from(
    new Set([...Object.keys(live), ...Object.keys(dbByClass), ...source.enabled_classes])
  );
  const labels = source.direction_labels ?? { in: "in", out: "out" };

  return (
    <div>
      <div className="mb-4 flex items-center gap-3">
        <h1 className="text-2xl font-semibold">{source.name}</h1>
        <StatusBadge status={source.status} />
        <span className="text-xs uppercase text-slate-500">{source.type}</span>
        <button
          onClick={toggle}
          className="ml-auto rounded bg-sky-600 px-4 py-2 text-sm font-medium hover:bg-sky-500"
        >
          {running ? "Stop" : "Start"}
        </button>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <div className="mb-3 flex gap-2">
            <button
              onClick={() => setTab("live")}
              className={`rounded px-3 py-1.5 text-sm ${tab === "live" ? "bg-sky-600" : "bg-slate-800"}`}
            >
              Live
            </button>
            <button
              onClick={() => setTab("line")}
              className={`rounded px-3 py-1.5 text-sm ${tab === "line" ? "bg-sky-600" : "bg-slate-800"}`}
            >
              Atur Garis Hitung
            </button>
          </div>

          {tab === "live" ? (
            <div className="overflow-hidden rounded-lg border border-slate-700 bg-black">
              {running ? (
                <img src={api.streamUrl(sourceId)} alt="live" className="w-full" />
              ) : (
                <div className="flex aspect-video items-center justify-center text-slate-500">
                  Source tidak berjalan. Klik Start untuk melihat live.
                </div>
              )}
            </div>
          ) : (
            <LineDrawCanvas source={source} onSaved={loadSource} />
          )}
        </div>

        <div className="space-y-6">
          <section className="rounded-lg border border-slate-800 bg-slate-900 p-4">
            <h2 className="mb-3 font-medium">Hitungan per Kelas</h2>
            {!source.line && (
              <p className="mb-2 text-xs text-amber-400">
                Garis hitung belum diatur — buka tab "Atur Garis Hitung".
              </p>
            )}
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-slate-400">
                  <th className="pb-1">Kelas</th>
                  <th className="pb-1">{labels.in}</th>
                  <th className="pb-1">{labels.out}</th>
                </tr>
              </thead>
              <tbody>
                {classes.map((c) => {
                  const v = (live[c] as any) ?? dbByClass[c] ?? { in: 0, out: 0 };
                  return (
                    <tr key={c} className="border-t border-slate-800">
                      <td className="py-1">{c}</td>
                      <td className="py-1 font-mono text-green-300">{v.in ?? 0}</td>
                      <td className="py-1 font-mono text-orange-300">{v.out ?? 0}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </section>

          <section className="rounded-lg border border-slate-800 bg-slate-900 p-4">
            <h2 className="mb-3 font-medium">Plat Nomor Terbaru</h2>
            {plates.length === 0 ? (
              <p className="text-sm text-slate-500">Belum ada plat terbaca.</p>
            ) : (
              <ul className="space-y-2">
                {plates.map((p) => (
                  <li key={p.id} className="flex items-center gap-3 text-sm">
                    {p.has_image && (
                      <img
                        src={api.plateImageUrl(p.id)}
                        alt={p.plate_text}
                        className="h-8 w-20 rounded border border-slate-700 object-cover"
                      />
                    )}
                    <span className="font-mono font-semibold tracking-wider">{p.plate_text}</span>
                    <span className="text-slate-500">{p.vehicle_class}</span>
                    <span className="ml-auto text-xs text-slate-500">
                      {new Date(p.timestamp).toLocaleTimeString()}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </section>

          {source.face_enabled && (
            <section className="rounded-lg border border-slate-800 bg-slate-900 p-4">
              <h2 className="mb-3 font-medium">Wajah Terdeteksi</h2>
              {sightings.length === 0 ? (
                <p className="text-sm text-slate-500">Belum ada wajah terdeteksi.</p>
              ) : (
                <ul className="space-y-2">
                  {sightings.map((s) => (
                    <li key={s.id} className="flex items-center gap-3 text-sm">
                      {s.has_image && (
                        <img
                          src={api.sightingImageUrl(s.id)}
                          alt={s.name ?? "unknown"}
                          className="h-10 w-10 rounded border border-slate-700 object-cover"
                        />
                      )}
                      <span className="font-medium">
                        {s.name ?? <span className="text-slate-500">unknown</span>}
                      </span>
                      {s.name && (
                        <span className="text-xs text-slate-500">
                          {(s.similarity * 100).toFixed(0)}%
                        </span>
                      )}
                      <span className="ml-auto text-xs text-slate-500">
                        {new Date(s.timestamp).toLocaleTimeString()}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
