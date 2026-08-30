import { ReactNode, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, DatasetSummary, OcrMetrics } from "../api";

// Below this many reviewed reads the percentages move several points every
// time somebody labels one more, and quoting them to two significant figures
// invites decisions they cannot support. The page still shows them -- watching
// the number settle is half the reason to keep reviewing -- but says so.
const STEADY_ENOUGH = 100;

export default function Dataset() {
  const [data, setData] = useState<DatasetSummary | null>(null);
  const [error, setError] = useState("");
  const [evalShare, setEvalShare] = useState(20);

  const load = () =>
    api
      .getDataset()
      .then((d) => {
        setData(d);
        setError("");
      })
      .catch((err: any) => setError(err?.message ?? "Gagal memuat data"));

  useEffect(() => {
    load();
  }, []);

  const p = data?.progress;
  const m = data?.metrics ?? null;
  const coverage = p && p.total > 0 ? p.reviewed / p.total : 0;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Akurasi OCR</h1>
        <button
          onClick={load}
          className="ml-auto rounded border border-slate-700 px-3 py-1.5 text-sm hover:bg-slate-800"
        >
          Muat ulang
        </button>
      </div>

      {error && (
        <p className="rounded border border-red-800 bg-red-500/10 p-3 text-sm text-red-300">
          {error}
        </p>
      )}

      {/* Coverage first, because every number under it is only as trustworthy
          as the sample it was measured on. */}
      <Card>
        <div className="flex flex-wrap items-baseline gap-3">
          <h2 className="text-lg font-semibold">Tinjauan</h2>
          <span className="text-sm text-slate-400">
            {p ? (
              <>
                <span className="font-mono text-slate-200">{p.reviewed.toLocaleString()}</span> dari{" "}
                <span className="font-mono">{p.total.toLocaleString()}</span> pembacaan sudah
                diperiksa manusia
              </>
            ) : (
              "memuat…"
            )}
          </span>
          <Link
            to="/plates?review=pending"
            className="ml-auto rounded border border-sky-600 bg-sky-500/20 px-3 py-1.5 text-sm text-sky-200 hover:bg-sky-500/30"
          >
            Tinjau yang belum diperiksa →
          </Link>
        </div>

        <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full rounded-full bg-sky-500 transition-all"
            style={{ width: `${Math.round(coverage * 100)}%` }}
          />
        </div>

        {p && (
          <div className="mt-3 flex flex-wrap gap-2 text-sm">
            <Tally label="OCR benar" n={p.correct} to="/plates?review=correct" tone="green" />
            <Tally label="OCR salah" n={p.wrong} to="/plates?review=wrong" tone="amber" />
            <Tally
              label="Tidak terbaca"
              n={p.illegible}
              to="/plates?review=illegible"
              tone="slate"
            />
            <Tally label="Belum ditinjau" n={p.pending} to="/plates?review=pending" tone="slate" />
          </div>
        )}
      </Card>

      {data && !m && (
        <Card>
          <h2 className="text-lg font-semibold">Belum ada yang bisa diukur</h2>
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400">
            Sistem ini tidak tahu seberapa akurat pembacaannya sendiri, dan tidak akan pernah tahu
            dari confidence OCR — angka itu menyatakan seberapa bersih decoder menyelesaikan satu
            gambar, bukan seberapa sering ia benar. Satu-satunya cara adalah seseorang melihat
            sebagian hasilnya dan mengatakan apa nomor yang sebenarnya.
          </p>
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400">
            Mulai dari{" "}
            <Link to="/plates?review=pending" className="text-sky-400 hover:underline">
              pembacaan yang belum ditinjau
            </Link>
            . Sekitar 100 sudah cukup untuk angka pertama yang berarti; tiap koreksi butuh beberapa
            detik, dan tekan Enter kalau OCR-nya memang sudah benar.
          </p>
        </Card>
      )}

      {m && (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            <Stat
              label="Terbaca persis"
              value={pct(m.exact_rate)}
              sub={`${m.exact} dari ${m.reads} pembacaan`}
              tone={m.exact_rate >= 0.8 ? "green" : m.exact_rate >= 0.5 ? "amber" : "red"}
            />
            <Stat
              label="Character error rate"
              value={pct(m.cer)}
              sub={`${m.errors} koreksi karakter atas ${m.chars} karakter`}
              tone={m.cer <= 0.05 ? "green" : m.cer <= 0.15 ? "amber" : "red"}
            />
            <Stat
              label="Tidak terbaca sama sekali"
              value={pct(m.blank_rate)}
              sub={`${m.blank} pembacaan kosong`}
              tone={m.blank_rate <= 0.05 ? "green" : "amber"}
            />
          </div>

          {m.reads < STEADY_ENOUGH && (
            <p className="rounded border border-amber-700 bg-amber-500/10 p-3 text-sm text-amber-200">
              Baru {m.reads} pembacaan yang ditinjau. Angka di atas masih bergerak beberapa poin
              tiap kali Anda menambah satu — pakai sebagai gambaran kasar, belum sebagai dasar
              keputusan. Sekitar {STEADY_ENOUGH} pembacaan sudah jauh lebih stabil.
            </p>
          )}

          {m.confusions.length > 0 && (
            <Card>
              <h2 className="text-lg font-semibold">Karakter yang paling sering tertukar</h2>
              <p className="mt-1 text-sm text-slate-400">
                Dihitung hanya dari salah-baca yang panjangnya sama, di mana pasangan karakternya
                tidak ambigu. Ini daftar yang seharusnya jadi dasar tabel perbaikan di{" "}
                <code className="rounded bg-slate-800 px-1 text-xs">alpr._CONFUSIONS</code> —
                terukur, bukan ditebak.
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {m.confusions.map((c) => (
                  <span
                    key={c.pair}
                    className="rounded border border-slate-700 bg-slate-800/60 px-2 py-1 font-mono text-sm"
                  >
                    {c.pair.replace("->", " → ")}
                    <span className="ml-2 text-xs text-slate-500">×{c.count}</span>
                  </span>
                ))}
              </div>
            </Card>
          )}

          {data && data.per_source.length > 1 && (
            <Card>
              <h2 className="text-lg font-semibold">Per kamera</h2>
              <p className="mt-1 text-sm text-slate-400">
                Terburuk di atas. Pembaca yang buruk jarang buruk merata — biasanya satu kamera
                yang terlalu tinggi, terlalu miring, atau menghadap matahari sore. Itu masalah yang
                diperbaiki dengan tangga, bukan dengan model.
              </p>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-slate-400">
                    <tr>
                      <th className="py-2 pr-3">Kamera</th>
                      <th className="py-2 pr-3">Ditinjau</th>
                      <th className="py-2 pr-3">Terbaca persis</th>
                      <th className="py-2 pr-3">CER</th>
                      <th className="py-2"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.per_source.map((s) => (
                      <tr key={s.source_id} className="border-t border-slate-800">
                        <td className="py-2 pr-3 text-slate-200">{s.name}</td>
                        <td className="py-2 pr-3 font-mono text-slate-400">{s.reads}</td>
                        <td className="py-2 pr-3 font-mono">{pct(s.exact_rate)}</td>
                        <td className="py-2 pr-3 font-mono text-slate-400">{pct(s.cer)}</td>
                        <td className="py-2">
                          <Link
                            to={`/plates?source=${s.source_id}&review=wrong`}
                            className="text-sky-400 hover:underline"
                          >
                            lihat yang salah
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}

      <Card>
        <h2 className="text-lg font-semibold">Ekspor dataset</h2>
        <p className="mt-1 max-w-3xl text-sm leading-relaxed text-slate-400">
          Satu file .zip berisi crop plat yang sudah ditinjau beserta labelnya — yang dibutuhkan
          untuk melatih pembaca khusus plat atau membandingkannya dengan yang sekarang. Pembacaan
          yang ditandai <em>tidak terbaca</em> tidak ikut: gambar yang tidak bisa dibaca siapa pun
          bukan contoh latih.
        </p>
        <ul className="mt-3 space-y-1 text-sm text-slate-400">
          <li>
            <code className="rounded bg-slate-800 px-1 text-xs">labels.csv</code> — kolom
            path,label; bentuk yang dibaca kebanyakan script training
          </li>
          <li>
            <code className="rounded bg-slate-800 px-1 text-xs">labels.jsonl</code> — baris yang
            sama plus prediksi OCR, confidence, source, waktu, dan peninjaunya
          </li>
          <li>
            <code className="rounded bg-slate-800 px-1 text-xs">train/</code> dan{" "}
            <code className="rounded bg-slate-800 px-1 text-xs">eval/</code> — dipisah{" "}
            <strong>per plat</strong>, bukan per baris: beberapa pembacaan satu mobil memakai plat
            yang sama, dan membiarkannya terbelah membocorkan jawaban ke set eval
          </li>
        </ul>

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <label htmlFor="eval-share" className="text-sm text-slate-400">
            Porsi eval
          </label>
          <input
            id="eval-share"
            type="number"
            min={0}
            max={90}
            step={5}
            value={evalShare}
            onChange={(e) => setEvalShare(Math.min(90, Math.max(0, Number(e.target.value) || 0)))}
            className="w-20 rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
          />
          <span className="text-sm text-slate-500">%</span>
          {m ? (
            <a
              href={api.datasetExportUrl(evalShare / 100)}
              className="rounded border border-sky-600 bg-sky-500/20 px-3 py-2 text-sm text-sky-200 hover:bg-sky-500/30"
            >
              Unduh dataset (.zip)
            </a>
          ) : (
            <span
              title="Belum ada pembacaan yang ditinjau, jadi belum ada yang bisa diekspor"
              className="rounded border border-slate-800 px-3 py-2 text-sm text-slate-600"
            >
              Unduh dataset (.zip)
            </span>
          )}
          {m && (
            <span className="text-sm text-slate-500">
              {m.reads.toLocaleString()} crop berlabel
            </span>
          )}
        </div>

        {data && data.skipped.length > 0 && (
          <p className="mt-3 text-sm text-slate-500">
            Tidak ikut diekspor:{" "}
            {data.skipped.map((s) => `${s.count} ${translateSkip(s.reason)}`).join(", ")}.
          </p>
        )}
      </Card>
    </div>
  );
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

// The reasons come from the backend in English, where they sit next to the
// code that produces them. The dashboard speaks Indonesian.
function translateSkip(reason: string): string {
  switch (reason) {
    case "reviewed as illegible":
      return "ditandai tidak terbaca";
    case "no crop recorded":
      return "tidak punya crop tersimpan";
    case "crop missing from disk":
      return "cropnya sudah hilang dari disk";
    default:
      return reason;
  }
}

function Card({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">{children}</div>
  );
}

const TONES: Record<string, string> = {
  green: "text-green-300",
  amber: "text-amber-300",
  red: "text-red-300",
  slate: "text-slate-300",
};

function Stat({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub: string;
  tone: keyof typeof TONES;
}) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 font-mono text-3xl font-semibold ${TONES[tone]}`}>{value}</div>
      <div className="mt-1 text-xs text-slate-500">{sub}</div>
    </div>
  );
}

function Tally({
  label,
  n,
  to,
  tone,
}: {
  label: string;
  n: number;
  to: string;
  tone: keyof typeof TONES;
}) {
  return (
    <Link
      to={to}
      className="rounded-full border border-slate-700 px-3 py-1 hover:border-slate-500 hover:bg-slate-800"
    >
      <span className="text-slate-400">{label}</span>{" "}
      <span className={`font-mono ${TONES[tone]}`}>{n.toLocaleString()}</span>
    </Link>
  );
}
