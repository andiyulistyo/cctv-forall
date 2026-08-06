import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, Source } from "../api";
import { StatusBadge } from "../components/common";
import AddSourceModal from "../components/AddSourceModal";

export default function Dashboard() {
  const [sources, setSources] = useState<Source[]>([]);
  const [active, setActive] = useState(0);
  const [showAdd, setShowAdd] = useState(false);
  const [error, setError] = useState("");

  const load = async () => {
    try {
      const r = await api.listSources();
      setSources(r.sources);
      setActive(r.active);
    } catch (err: any) {
      setError(err.message);
    }
  };

  useEffect(() => {
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  const toggle = async (s: Source) => {
    if (s.status === "running" || s.status === "starting") await api.stopSource(s.id);
    else await api.startSource(s.id);
    load();
  };

  const remove = async (s: Source) => {
    if (!confirm(`Hapus source "${s.name}"?`)) return;
    await api.deleteSource(s.id);
    load();
  };

  return (
    <div>
      <div className="mb-6 flex items-center gap-4">
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <span className="rounded-full bg-slate-800 px-3 py-1 text-sm">
          <span className="font-semibold text-green-400">{active}</span> aktif /{" "}
          {sources.length} source
        </span>
        <button
          onClick={() => setShowAdd(true)}
          className="ml-auto rounded bg-sky-600 px-4 py-2 font-medium hover:bg-sky-500"
        >
          + Tambah Source
        </button>
      </div>

      {error && <div className="mb-4 text-sm text-red-400">{error}</div>}

      {sources.length === 0 ? (
        <div className="rounded-xl border border-dashed border-slate-700 p-12 text-center text-slate-400">
          Belum ada source. Klik "Tambah Source" untuk mulai.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {sources.map((s) => (
            <div key={s.id} className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900">
              <Link to={`/sources/${s.id}`} className="block bg-black">
                {s.status === "running" ? (
                  <img
                    src={api.streamUrl(s.id)}
                    alt={s.name}
                    className="aspect-video w-full object-cover"
                  />
                ) : (
                  <div className="flex aspect-video w-full items-center justify-center text-slate-600">
                    {s.status === "error" ? "⚠ error" : "offline"}
                  </div>
                )}
              </Link>
              <div className="p-4">
                <div className="flex items-center gap-2">
                  <Link to={`/sources/${s.id}`} className="font-medium hover:text-sky-300">
                    {s.name}
                  </Link>
                  <StatusBadge status={s.status} />
                  {!s.line && (
                    <span className="text-xs text-amber-400" title="Garis hitung belum diatur">
                      · garis?
                    </span>
                  )}
                </div>
                <div className="mt-1 text-xs uppercase text-slate-500">{s.type}</div>
                <div className="mt-1 flex flex-wrap gap-1">
                  {s.enabled_classes.map((c) => (
                    <span key={c} className="rounded bg-slate-800 px-1.5 py-0.5 text-[11px] text-slate-300">
                      {c}
                    </span>
                  ))}
                </div>
                {s.status_message && (
                  <div className="mt-2 truncate text-xs text-red-400" title={s.status_message}>
                    {s.status_message}
                  </div>
                )}
                <div className="mt-3 flex gap-2">
                  <button
                    onClick={() => toggle(s)}
                    className="flex-1 rounded bg-slate-800 px-3 py-1.5 text-sm hover:bg-slate-700"
                  >
                    {s.status === "running" || s.status === "starting" ? "Stop" : "Start"}
                  </button>
                  <Link
                    to={`/sources/${s.id}`}
                    className="rounded bg-slate-800 px-3 py-1.5 text-sm hover:bg-slate-700"
                  >
                    Detail
                  </Link>
                  <button
                    onClick={() => remove(s)}
                    className="rounded bg-red-900/50 px-3 py-1.5 text-sm text-red-300 hover:bg-red-900"
                  >
                    Hapus
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {showAdd && <AddSourceModal onClose={() => setShowAdd(false)} onCreated={load} />}
    </div>
  );
}
