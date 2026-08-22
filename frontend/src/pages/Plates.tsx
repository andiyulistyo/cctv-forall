import { ReactNode, useEffect, useState } from "react";
import { api, Plate, Source } from "../api";
import { PlateEvidenceModal } from "../components/common";

const PAGE_SIZE = 15;

export default function Plates() {
  const [plates, setPlates] = useState<Plate[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [sources, setSources] = useState<Source[]>([]);
  const [sourceId, setSourceId] = useState<number | undefined>(undefined);
  const [evidence, setEvidence] = useState<Plate | null>(null);

  useEffect(() => {
    api.listSources().then((r) => setSources(r.sources));
  }, []);

  useEffect(() => {
    const load = async () => {
      const r = await api.listPlates(sourceId, PAGE_SIZE, page * PAGE_SIZE);
      setPlates(r.plates);
      setTotal(r.total);
    };
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [sourceId, page]);

  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  // Retention can delete rows out from under us; without this, a page that no
  // longer exists would just render empty forever.
  useEffect(() => {
    if (page > 0 && page >= pageCount) setPage(pageCount - 1);
  }, [page, pageCount]);

  const pickSource = (v: string) => {
    setSourceId(v ? Number(v) : undefined);
    setPage(0); // page 3 of "all sources" means nothing once a source is picked
  };

  const nameOf = (id: number) => sources.find((s) => s.id === id)?.name ?? `#${id}`;

  const first = total === 0 ? 0 : page * PAGE_SIZE + 1;
  const last = Math.min(total, page * PAGE_SIZE + plates.length);

  return (
    <div>
      <PlateEvidenceModal plate={evidence} onClose={() => setEvidence(null)} />
      <div className="mb-5 flex items-center gap-3">
        <h1 className="text-2xl font-semibold">Plat Nomor</h1>
        <select
          className="ml-auto rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
          value={sourceId ?? ""}
          onChange={(e) => pickSource(e.target.value)}
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

      <Pagination page={page} pageCount={pageCount} onPage={setPage}>
        {total === 0 ? "0 data" : `Menampilkan ${first}–${last} dari ${total} data`}
      </Pagination>
    </div>
  );
}

/** Page numbers with an ellipsis, so a few thousand plates stay one line wide. */
function Pagination({
  page,
  pageCount,
  onPage,
  children,
}: {
  page: number;
  pageCount: number;
  onPage: (p: number) => void;
  children: ReactNode;
}) {
  const btn = "rounded border border-slate-700 px-3 py-1.5 text-sm disabled:opacity-40";
  return (
    <div className="mt-4 flex flex-wrap items-center gap-2">
      <span className="text-sm text-slate-500">{children}</span>
      <div className="ml-auto flex items-center gap-1">
        <button className={btn} onClick={() => onPage(page - 1)} disabled={page === 0}>
          Sebelumnya
        </button>
        {pageNumbers(page, pageCount).map((n, i) =>
          n === null ? (
            <span key={`gap${i}`} className="px-1 text-slate-600">
              …
            </span>
          ) : (
            <button
              key={n}
              onClick={() => onPage(n)}
              className={`rounded px-3 py-1.5 text-sm ${
                n === page ? "bg-sky-600" : "border border-slate-700 hover:bg-slate-800"
              }`}
            >
              {n + 1}
            </button>
          )
        )}
        <button
          className={btn}
          onClick={() => onPage(page + 1)}
          disabled={page >= pageCount - 1}
        >
          Berikutnya
        </button>
      </div>
    </div>
  );
}

/** Always the first and last page, plus a window around the current one.
 *  null means "gap" and renders as an ellipsis. */
function pageNumbers(page: number, pageCount: number): (number | null)[] {
  if (pageCount <= 7) return Array.from({ length: pageCount }, (_, i) => i);
  const window = new Set([0, pageCount - 1, page, page - 1, page + 1]);
  // Keep the row a constant width: pad the window when we sit at either end.
  if (page <= 2) [1, 2, 3].forEach((n) => window.add(n));
  if (page >= pageCount - 3) [pageCount - 4, pageCount - 3, pageCount - 2].forEach((n) => window.add(n));

  const pages = [...window].filter((n) => n >= 0 && n < pageCount).sort((a, b) => a - b);
  const out: (number | null)[] = [];
  pages.forEach((n, i) => {
    if (i > 0 && n - pages[i - 1] > 1) out.push(null);
    out.push(n);
  });
  return out;
}
