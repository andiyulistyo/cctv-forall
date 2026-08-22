import { ReactNode, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  api,
  CLASS_LABELS,
  Plate,
  PLATE_SORT_KEYS,
  PlateSortKey,
  Source,
  VEHICLE_CLASSES,
} from "../api";
import { PlateEvidenceModal } from "../components/common";

const PAGE_SIZE = 15;

// Which way each column wants to be read the first time you click it: newest
// reads and best matches first, but names and plates alphabetically.
const DEFAULT_ORDER: Record<PlateSortKey, "asc" | "desc"> = {
  timestamp: "desc",
  confidence: "desc",
  plate_text: "asc",
  vehicle_class: "asc",
  source: "asc",
};

const RANGES = [
  { key: "", label: "Semua waktu", ms: 0 },
  { key: "1h", label: "1 jam terakhir", ms: 3_600_000 },
  { key: "24h", label: "24 jam terakhir", ms: 86_400_000 },
  { key: "7d", label: "7 hari terakhir", ms: 604_800_000 },
];

const CONFIDENCES = [
  { key: "", label: "Semua confidence" },
  { key: "0.5", label: "Confidence ≥ 50%" },
  { key: "0.7", label: "Confidence ≥ 70%" },
  { key: "0.9", label: "Confidence ≥ 90%" },
];

export default function Plates() {
  const [plates, setPlates] = useState<Plate[]>([]);
  const [total, setTotal] = useState(0);
  const [classCounts, setClassCounts] = useState<Record<string, number>>({});
  const [sources, setSources] = useState<Source[]>([]);
  const [evidence, setEvidence] = useState<Plate | null>(null);
  const [error, setError] = useState("");
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  // Rows shift under the cursor every five seconds, which makes comparing two
  // reads a race. Pausing freezes the table without losing the filters.
  const [live, setLive] = useState(true);
  const [reloads, setReloads] = useState(0);

  // Every filter lives in the URL: a view worth looking at is worth sending to
  // someone else, and a refresh should not throw it away.
  const [params, setParams] = useSearchParams();
  const sourceId = params.get("source") ? Number(params.get("source")) : undefined;
  const vehicleClass = params.get("class") ?? "";
  const minConfidence = params.get("conf") ?? "";
  const range = params.get("range") ?? "";
  const urlQ = params.get("q") ?? "";
  const page = Math.max(0, Number(params.get("page") ?? 0) || 0);
  // Hand-edited URLs are the one place these can arrive as nonsense, and the
  // server answers nonsense with a 422 rather than a page of plates.
  const rawSort = params.get("sort") as PlateSortKey;
  const sort: PlateSortKey = PLATE_SORT_KEYS.includes(rawSort) ? rawSort : "timestamp";
  const order = params.get("order") === "asc" ? "asc" : "desc";

  const patch = (next: Record<string, string | undefined>, keepPage = false) => {
    const p = new URLSearchParams(params);
    Object.entries(next).forEach(([k, v]) => (v ? p.set(k, v) : p.delete(k)));
    if (!keepPage) p.delete("page"); // page 3 of the old filter means nothing under the new one
    setParams(p, { replace: true });
  };

  const filtered = Boolean(sourceId || vehicleClass || minConfidence || range || urlQ);
  const resetFilters = () =>
    setParams(sort === "timestamp" && order === "desc" ? {} : { sort, order }, { replace: true });

  // The box has to follow the URL, not just drive it: Reset and the back
  // button both rewrite `q` from outside this input.
  const [qInput, setQInput] = useState(urlQ);
  useEffect(() => setQInput(urlQ), [urlQ]);
  useEffect(() => {
    if (qInput === urlQ) return;
    const t = setTimeout(() => patch({ q: qInput }), 350); // don't refetch per keystroke
    return () => clearTimeout(t);
  }, [qInput, urlQ]);

  useEffect(() => {
    api.listSources().then((r) => setSources(r.sources)).catch(() => setSources([]));
  }, []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      // Recomputed per fetch, so "24 jam terakhir" keeps meaning the last 24
      // hours rather than the 24 hours before the page happened to load.
      const ms = RANGES.find((r) => r.key === range)?.ms ?? 0;
      try {
        const r = await api.listPlates({
          source: sourceId,
          vehicle_class: vehicleClass || undefined,
          q: urlQ || undefined,
          min_confidence: minConfidence ? Number(minConfidence) : undefined,
          from: ms ? new Date(Date.now() - ms).toISOString() : undefined,
          sort,
          order,
          limit: PAGE_SIZE,
          offset: page * PAGE_SIZE,
        });
        if (cancelled) return;
        setPlates(r.plates);
        setTotal(r.total);
        setClassCounts(r.class_counts);
        setUpdatedAt(new Date());
        setError("");
      } catch (err: any) {
        if (!cancelled) setError(err?.message ?? "Gagal memuat data");
      }
    };
    load();
    if (!live) return () => void (cancelled = true);
    const t = setInterval(load, 5000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [sourceId, vehicleClass, urlQ, minConfidence, range, sort, order, page, live, reloads]);

  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  // Retention can delete rows out from under us; without this, a page that no
  // longer exists would just render empty forever.
  useEffect(() => {
    if (page > 0 && page >= pageCount) patch({ page: String(pageCount - 1) }, true);
  }, [page, pageCount]);

  const nameOf = (id: number) => sources.find((s) => s.id === id)?.name ?? `#${id}`;

  const onSort = (col: PlateSortKey) =>
    patch(
      col === sort
        ? { sort: col, order: order === "asc" ? "desc" : "asc" }
        : { sort: col, order: DEFAULT_ORDER[col] },
      true // re-sorting is not re-filtering; the row you were near is still in the set
    );

  const classTotal = useMemo(
    () => Object.values(classCounts).reduce((a, b) => a + b, 0),
    [classCounts]
  );

  const first = total === 0 ? 0 : page * PAGE_SIZE + 1;
  const last = Math.min(total, page * PAGE_SIZE + plates.length);
  const needle = urlQ.replace(/[^a-z0-9]/gi, "");

  return (
    <div>
      <PlateEvidenceModal plate={evidence} onClose={() => setEvidence(null)} />

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Plat Nomor</h1>
        <div className="ml-auto flex items-center gap-2 text-sm">
          {updatedAt && (
            <span className="text-slate-500">
              Diperbarui {updatedAt.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={() => setLive(!live)}
            title={live ? "Hentikan pembaruan otomatis" : "Lanjutkan pembaruan tiap 5 detik"}
            className={`rounded border px-3 py-1.5 ${
              live
                ? "border-green-700 bg-green-500/10 text-green-300"
                : "border-slate-700 text-slate-400"
            }`}
          >
            {live ? "● Live" : "❚❚ Jeda"}
          </button>
          {!live && (
            <button
              onClick={() => setReloads((n) => n + 1)}
              className="rounded border border-slate-700 px-3 py-1.5 hover:bg-slate-800"
            >
              Muat ulang
            </button>
          )}
        </div>
      </div>

      <div className="mb-4 space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={qInput}
            onChange={(e) => setQInput(e.target.value)}
            placeholder="Cari nomor plat…"
            aria-label="Cari nomor plat"
            className="w-56 rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm placeholder:text-slate-500"
          />
          <Select
            value={sourceId ?? ""}
            onChange={(v) => patch({ source: v })}
            aria-label="Filter source"
          >
            <option value="">Semua source</option>
            {sources.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </Select>
          <Select value={range} onChange={(v) => patch({ range: v })} aria-label="Filter waktu">
            {RANGES.map((r) => (
              <option key={r.key} value={r.key}>
                {r.label}
              </option>
            ))}
          </Select>
          <Select
            value={minConfidence}
            onChange={(v) => patch({ conf: v })}
            aria-label="Filter confidence"
          >
            {CONFIDENCES.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </Select>
          <Select
            value={`${sort}:${order}`}
            onChange={(v) => {
              const [s, o] = v.split(":");
              patch({ sort: s, order: o }, true);
            }}
            aria-label="Urutkan"
          >
            <option value="timestamp:desc">Urut: Terbaru</option>
            <option value="timestamp:asc">Urut: Terlama</option>
            <option value="confidence:desc">Urut: Confidence tertinggi</option>
            <option value="confidence:asc">Urut: Confidence terendah</option>
            <option value="plate_text:asc">Urut: Nomor A→Z</option>
            <option value="plate_text:desc">Urut: Nomor Z→A</option>
            <option value="vehicle_class:asc">Urut: Kelas A→Z</option>
            <option value="source:asc">Urut: Source A→Z</option>
          </Select>
          {filtered && (
            <button
              onClick={resetFilters}
              className="rounded border border-slate-700 px-3 py-2 text-sm text-slate-400 hover:bg-slate-800 hover:text-slate-200"
            >
              Reset filter
            </button>
          )}
        </div>

        {/* Counts come from the server under every filter except this one, so
            they say what each class would give you instead of dropping to zero. */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs uppercase tracking-wide text-slate-500">Kelas</span>
          <Chip active={!vehicleClass} onClick={() => patch({ class: "" })}>
            Semua <Count n={classTotal} />
          </Chip>
          {VEHICLE_CLASSES.map((c) => {
            const n = classCounts[c] ?? 0;
            return (
              <Chip
                key={c}
                active={vehicleClass === c}
                muted={n === 0}
                onClick={() => patch({ class: vehicleClass === c ? "" : c })}
              >
                {CLASS_LABELS[c]} <Count n={n} />
              </Chip>
            );
          })}
        </div>
      </div>

      {error && (
        <p className="mb-3 rounded border border-red-800 bg-red-500/10 p-3 text-sm text-red-300">
          {error}
        </p>
      )}

      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table className="w-full text-sm">
          <thead className="bg-slate-900 text-left text-slate-400">
            <tr>
              <th className="p-3">Plat (crop)</th>
              <th className="p-3">Kendaraan</th>
              <SortHeader col="plate_text" sort={sort} order={order} onSort={onSort}>
                Nomor
              </SortHeader>
              <SortHeader col="vehicle_class" sort={sort} order={order} onSort={onSort}>
                Kelas
              </SortHeader>
              <SortHeader col="source" sort={sort} order={order} onSort={onSort}>
                Source
              </SortHeader>
              <SortHeader col="confidence" sort={sort} order={order} onSort={onSort}>
                Confidence
              </SortHeader>
              <SortHeader col="timestamp" sort={sort} order={order} onSort={onSort}>
                Waktu
              </SortHeader>
            </tr>
          </thead>
          <tbody>
            {plates.map((p) => (
              <tr key={p.id} className="border-t border-slate-800 hover:bg-slate-900/50">
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
                  <Highlight text={p.plate_text} needle={needle} />
                </td>
                <td className="p-3">
                  <ClassBadge name={p.vehicle_class} />
                </td>
                <td className="p-3 text-slate-300">{nameOf(p.source_id)}</td>
                <td className="p-3">
                  <ConfidenceBadge value={p.confidence} />
                </td>
                <td className="p-3 text-slate-400" title={new Date(p.timestamp).toLocaleString()}>
                  {relativeTime(p.timestamp)}
                </td>
              </tr>
            ))}
            {plates.length === 0 && (
              <tr>
                <td colSpan={7} className="p-8 text-center text-slate-500">
                  {filtered ? (
                    <>
                      Tidak ada plat yang cocok dengan filter ini.{" "}
                      <button onClick={resetFilters} className="text-sky-400 hover:underline">
                        Reset filter
                      </button>
                    </>
                  ) : (
                    "Belum ada data plat nomor."
                  )}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <Pagination page={page} pageCount={pageCount} onPage={(p) => patch({ page: String(p) }, true)}>
        {total === 0 ? "0 data" : `Menampilkan ${first}–${last} dari ${total} data`}
        {filtered && total > 0 && " (terfilter)"}
      </Pagination>
    </div>
  );
}

function Select({
  value,
  onChange,
  children,
  ...rest
}: {
  value: string | number;
  onChange: (v: string) => void;
  children: ReactNode;
} & { "aria-label"?: string }) {
  return (
    <select
      {...rest}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
    >
      {children}
    </select>
  );
}

function Chip({
  active,
  muted,
  onClick,
  children,
}: {
  active: boolean;
  muted?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={`rounded-full border px-3 py-1 text-sm ${
        active
          ? "border-sky-500 bg-sky-500/20 text-sky-200"
          : muted
            ? "border-slate-800 text-slate-600"
            : "border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-200"
      }`}
    >
      {children}
    </button>
  );
}

function Count({ n }: { n: number }) {
  return <span className="ml-1 font-mono text-xs opacity-70">{n}</span>;
}

/** A column header that sorts, and shows which way it is sorting. */
function SortHeader({
  col,
  sort,
  order,
  onSort,
  children,
}: {
  col: PlateSortKey;
  sort: PlateSortKey;
  order: "asc" | "desc";
  onSort: (c: PlateSortKey) => void;
  children: ReactNode;
}) {
  const active = sort === col;
  return (
    <th className="p-3" aria-sort={active ? (order === "asc" ? "ascending" : "descending") : "none"}>
      <button
        onClick={() => onSort(col)}
        title={active ? "Klik untuk membalik urutan" : "Urutkan kolom ini"}
        className={`group inline-flex items-center gap-1 ${
          active ? "text-sky-300" : "hover:text-slate-200"
        }`}
      >
        {children}
        {/* Kept in the layout even when inactive so headers don't jump on hover. */}
        <span className={active ? "" : "opacity-0 group-hover:opacity-40"}>
          {active && order === "asc" ? "▲" : "▼"}
        </span>
      </button>
    </th>
  );
}

const CLASS_STYLES: Record<string, string> = {
  car: "bg-sky-500/15 text-sky-300 border-sky-700",
  motorcycle: "bg-violet-500/15 text-violet-300 border-violet-700",
  truck: "bg-amber-500/15 text-amber-300 border-amber-700",
  bus: "bg-emerald-500/15 text-emerald-300 border-emerald-700",
};

function ClassBadge({ name }: { name: string }) {
  const cls = CLASS_STYLES[name] ?? "bg-slate-700/30 text-slate-300 border-slate-600";
  return (
    <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`} title={name}>
      {CLASS_LABELS[name] ?? name}
    </span>
  );
}

/** A weak read is the one you most need to eyeball against the crop, so it is
 *  the one the colour has to call out. */
function ConfidenceBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const cls =
    value >= 0.8
      ? "text-green-300"
      : value >= 0.6
        ? "text-amber-300"
        : "text-red-300";
  return (
    <span className="inline-flex items-center gap-2" title={`OCR confidence ${pct}%`}>
      <span className="h-1.5 w-14 overflow-hidden rounded-full bg-slate-800">
        <span
          className={`block h-full rounded-full ${
            value >= 0.8 ? "bg-green-500" : value >= 0.6 ? "bg-amber-500" : "bg-red-500"
          }`}
          style={{ width: `${Math.max(pct, 2)}%` }}
        />
      </span>
      <span className={`font-mono text-xs ${cls}`}>{pct}%</span>
    </span>
  );
}

/** Marks the searched-for characters inside the plate, so it is obvious why a
 *  row matched. The server ignores spaces and dashes, so we do too. */
function Highlight({ text, needle }: { text: string; needle: string }) {
  if (!needle) return <>{text}</>;
  const at = text.toLowerCase().indexOf(needle.toLowerCase());
  if (at < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, at)}
      <mark className="rounded bg-sky-500/30 text-sky-100">{text.slice(at, at + needle.length)}</mark>
      {text.slice(at + needle.length)}
    </>
  );
}

/** "3 mnt lalu" reads faster than a timestamp when you are watching a gate.
 *  Anything older than a day is a date again -- "37 jam lalu" helps nobody. */
function relativeTime(iso: string): string {
  const then = new Date(iso);
  const secs = (Date.now() - then.getTime()) / 1000;
  if (secs < 45) return "baru saja";
  if (secs < 3600) return `${Math.round(secs / 60)} mnt lalu`;
  if (secs < 86400) return `${Math.round(secs / 3600)} jam lalu`;
  return then.toLocaleString();
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
