# Rencana kerja: satu baris per kendaraan, frame terbaik, plat terbaca

Basis: commit `7b66c13`, branch `optimize`. Semua angka di dokumen ini diukur
dari `data/app.db` (2945 baris, 23–30 Agustus 2026, source 6 & 7) dan dari file
gambar nyata di `data/plates/` + `data/frames/`. Bukan perkiraan.

## Tujuan

1. **Satu objek = satu baris per source.** Sekarang listing `/plates` berisi
   ±1,28 baris per kendaraan nyata.
2. **Objek hanya boleh di-capture saat berada di dalam zona ANPR.** Berlaku untuk
   jalur capture maupun jalur read.
3. **Frame yang disimpan harus frame terbaik selama objek di dalam zona**, supaya
   pembacaan plat seakurat mungkin.
4. **Pembacaan plat lebih baik untuk semua kelas**: motor, mobil, truk.

---

## Cara pakai dokumen ini

**Untuk saya (Andi):** koreksi langsung di file ini. Yang paling perlu keputusan
ada di bagian [Keputusan yang menunggu](#keputusan-yang-menunggu) — beberapa task
sengaja saya tandai `BLOCKED` sampai keputusan itu diambil.

**Untuk agent yang menjalankan:**

- **Cari kode lewat nama simbol, bukan nomor baris.** Nomor baris di dokumen ini
  benar per `7b66c13` dan akan bergeser begitu task pertama diterapkan.
- **Jalankan tes** dari direktori `backend/`. Repo ini tidak memakai pytest —
  tiap modul tes berdiri sendiri:
  ```
  cd backend && ./.venv/Scripts/python.exe -m tests.test_plate_ocr
  ```
  Sebelum menyatakan selesai, jalankan **semua** modul di `backend/tests/`.
- **Database `data/app.db` hanya boleh dibuka read-only** (`file:data/app.db?mode=ro`,
  `uri=True`). Backend sedang berjalan.
- **Jangan sentuh `.env`.** Setting baru ditulis ke `backend/app/config.py` dengan
  default yang aman, dan didokumentasikan di `.env.nvidia.example`.
- **Jangan push.** Satu commit per task, mengikuti gaya repo: subjek satu kalimat
  perilaku, badan prosa yang menjelaskan *kenapa* berikut angka yang mengukurnya.
- **Setiap perubahan perilaku wajib punya tes** di `backend/tests/`, mengikuti pola
  file yang sudah ada (stdlib, `python -m tests.x`, cetak `OK <nama>`).
- Kalau sebuah task ternyata salah premis saat dikerjakan — **berhenti dan lapor**,
  jangan diakali. Beberapa task di sini ditulis dari pengukuran, bukan dari
  membaca seluruh kode.

---

## Baseline — angka yang harus dikalahkan

Bekukan ini sebelum mengubah apa pun (lihat T0.1). Era kode sekarang = baris
dengan `timestamp >= 2026-08-27`.

**Pembacaan plat**

```
                 rows   non-blank      usable (>=7 char)
motorcycle       1110   335 (30,2%)    159 (14,3%)
car               484   484 (100%)     347 (71,7%)
truck              41    41 (100%)      23 (56,1%)
```

Dari 138 baris yang sudah ditinjau manusia (108 legible, sisanya `corrected_text=''`):

```
ALL         n=108   exact 14 (13,0%)   CER 44,2%
car         n= 56   exact 12 (21,4%)   CER 32,7%
motorcycle  n= 50   exact  1 ( 2,0%)   CER 58,5%   11 baris kosong padahal terbaca mata
```

Panjang prediksi vs kebenaran: motor **4,74 vs 7,80** karakter. Anggaran error
motor: **154 penghapusan : 73 substitusi : 1 penyisipan**. Exact-match per panjang
prediksi: `len4 → 0/20, len5 → 0/9, len6 → 0/9, len7 → 1/22, len8 → 13/32`.

**Duplikat** (diklaster dengan jam kamera, bukan `timestamp` DB)

```
513 klaster (>=2 baris, satu source, jarak <=8 detik) = 1247 baris = 42% listing
±640 baris berlebih (22%)  ->  ±2300 baris tersisa  ->  1,28 baris per lintasan
```

| Penyebab | Baris | % |
|---|---:|---:|
| (a) tracker menomori ulang objek bergerak | ~555 | 18,8% |
| (b) deteksi terpecah/bersarang dalam 1 frame | 88 | 3,0% |
| (c) kelas berubah pada satu objek | 77 | 2,6% |
| (d) teks OCR berbeda tipis, lolos jendela 60s | ~47 | 1,6% |
| (e) **kendaraan berbeda sungguhan — bukan duplikat** | ~150 | 12% dari klaster |

**Fakta lain yang jadi dasar rencana ini**

- `plate_reads.timestamp` adalah waktu **baris ditulis**, bukan waktu frame. Lag
  median **2,0 detik**, p95 8 detik, maks 266 detik.
- 482 dari 1064 pasang duplikat ber-IoU **tepat 0,00** — kotaknya tidak bersentuhan.
- 1625 dari 1755 baris motor kosong (**92,6%**) tidak punya plat yang ditemukan
  sama sekali; yang tersimpan crop motor utuh (median 225×350, aspek 0,64).
- Dalam satu lintasan, kotak terlebar **median 1,92×** kotak tersempit; >2× pada 47%.
- Confidence OCR tidak terkalibrasi: band 0,6–0,8 → **0% exact** (n=31); 0,95+ → 36%.
- Detektor plat: **78,7 ms/panggil** dengan crop beraneka bentuk vs **14,6 ms**
  kalau di-letterbox ke 320×320 konstan (174,6 ms pada crop 4K).
- Malam: motor **1,6% usable** (4 plat dari 245 lintasan), mobil 40,7%.

---

## Peta konflik file

Untuk menjalankan beberapa agent paralel. **Satu file = satu agent pada satu
waktu.** Task di kolom yang sama harus berurutan; kolom berbeda boleh paralel.

| File | Task |
|---|---|
| `backend/app/detection/alpr.py` | T1.1 T1.2 T1.3 T1.4 T1.5 · T5.1 T5.3 |
| `backend/app/detection/worker.py` | T2.1 T2.2 T2.4 · T3.1 T3.2 T3.3 · T4.1 T4.2 T4.3 · T5.2 |
| `backend/app/detection/capture_gate.py` | T2.3 · T3.2 |
| `backend/app/detection/vehicle_registry.py` | T2.5 · T3.4 · T4.1 |
| `backend/app/models.py` + `database.py` | T3.1 |
| `backend/app/config.py` | hampir semua gelombang menambah setting di sini — **milik agent terakhir dalam gelombang**, atau selesaikan konflik saat merge |
| `scripts/` | T0.1 T0.2 |

Saran: satu gelombang = satu worktree, lalu merge. Gelombang 1 dan Gelombang 2
tidak saling bergantung dan boleh jalan bersamaan.

---

## Gelombang 0 — ukur dulu (read-only, boleh paralel penuh)

### T0.1 — Skrip audit ANPR yang bisa diulang
- [ ] **File:** `scripts/anpr_audit.py` (baru)
- **Isi:** buka `data/app.db` read-only, cetak (a) blank/non-blank/usable per kelas
  untuk rentang tanggal yang diberikan, (b) exact-match + CER per kelas dari baris
  ter-review, (c) distribusi panjang prediksi vs kebenaran, (d) taksonomi klaster
  duplikat. Terima `--since` dan `--source`.
- **Kenapa:** setiap task sesudah ini mengklaim perbaikan. Tanpa satu skrip yang
  memberi angka yang sama sebelum dan sesudah, klaim itu tidak bisa diperiksa —
  dan 138 review yang ada dikumpulkan **di bawah scorer yang sedang regresi**
  (lihat T1.3), jadi angkanya akan bergeser sendiri.
- **Catatan:** aritmetikanya sedapat mungkin dipinjam dari `app/plate_dataset.py`
  supaya halaman *Akurasi OCR* dan skrip ini tidak pernah berbeda angka.
- **Selesai kalau:** dijalankan tanpa argumen mereproduksi tabel baseline di atas
  dalam ±1 baris.
- **Risiko:** rendah, tidak menyentuh jalur produksi.

### T0.2 — Validasi offline: PLATE_IMGSZ 320 vs 640 pada crop motor yang gagal
- [ ] **File:** `scripts/` (skrip sekali pakai, boleh di scratchpad — yang penting angkanya)
- **Isi:** ambil **1625 crop motor utuh** di `data/plates/` (baris blank yang
  `locate_plate`-nya mengembalikan None — kenali dari aspek < 1,5). Jalankan
  `license-plate-finetune-v1s.pt` pada tiap crop di `imgsz=320` dan `imgsz=640`,
  dan juga pada crop yang di-upscale 2× dan 3× sebelum masuk detektor. Hitung:
  berapa persen yang akhirnya menemukan kotak plat, dan berapa lebar kotaknya.
- **Kenapa:** ini menentukan apakah T5.1 layak dikerjakan sama sekali. Datanya
  sudah ada di disk — tidak perlu menunggu lalu lintas baru dan tidak perlu
  menyentuh sistem yang berjalan.
- **Selesai kalau:** ada tabel `(imgsz, upscale) → % plat ditemukan, median lebar
  kotak plat, ms/panggil` ditulis ke dokumen ini di bawah T5.1.
- **Risiko:** memakai GPU yang sedang dipakai dua worker 4K. Jalankan saat sepi,
  atau `device=cpu` dan abaikan angka ms-nya.

---

## Gelombang 1 — akurasi baca, semua kelas

Satu file (`alpr.py`), jadi satu agent, berurutan. Ini gelombang dengan rasio
untung/usaha tertinggi dan **membalik regresi mobil**.

### T1.1 — Naikkan lantai bentuk plat ke 7 karakter / 3 digit
- [ ] **File:** `backend/app/detection/alpr.py`
- **Anchor:** `_PLATE_RE`, `looks_like_plate`, `_MIN_PLATE_DIGITS`
- **Ubah:** `_PLATE_RE` dari `^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$` menjadi
  `^[A-Z]{1,2}\d{3,4}[A-Z]{1,3}$`; `looks_like_plate` dari `4 <= len <= 9`
  menjadi `7 <= len <= 9`. Jadikan angka minimumnya sebuah setting
  (`ALPR_MIN_PLATE_CHARS`, default 7) supaya bisa dikembalikan tanpa deploy ulang.
- **Kenapa:** **setiap bacaan di bawah 7 karakter salah — 0 dari 38.** Diukur pada
  97 bacaan non-blank yang ditinjau: **CER 37,7% → 19,1%, exact 14,4% → 23,7%**;
  per kelas, mobil 32,7% → 19,4%, motor 58,5% → 18,9%. Bukti nyata di disk:
  `data/plates/7_1143_20260824_073651_C8JC.jpg` adalah crop plat bersih yang jelas
  terbaca **B 4763 UFT**; yang tersimpan `C8JC` @0.647 — terbaca dari baris masa
  berlaku di bawah nomornya.
- **Dukungan dari kode sendiri:** komentar `_MIN_PLATE_DIGITS` di `_candidates`
  sudah menuliskan premisnya — *"a road-going Indonesian plate carries three or
  four"* digit. Lantai 3 digit di T1.1 hanyalah menegakkan apa yang komentar itu
  sudah nyatakan; yang sekarang berlaku baru 2.
- **Selesai kalau:** tes baru di `test_plate_ocr.py` memastikan `C8JC`, `Z71I`,
  `O62G`, `S50O`, `G96B` ditolak dan `B4763UFT`, `B6084TXB`, `AD1234AB` diterima;
  seluruh modul tes lain tetap lulus.
- **Risiko:** **baris yang tadinya "ada isinya" jadi kosong — secara persepsi
  terlihat mundur**, padahal yang hilang 0/38 benar. Ini konsekuensi yang disengaja,
  tulis di badan commit-nya. Plat lama berformat 1–2 digit (langka, biasanya
  kendaraan dinas) akan ikut tertolak — kalau di lokasi ini ada, turunkan ke 6.

### T1.2 — Fragmen tidak boleh menghentikan tangga preprocessing
- [ ] **File:** `backend/app/detection/alpr.py`
- **Anchor:** fungsi `consider()` di dalam `read_plate`, baris `if conf >= self.good_enough: strong = True`
- **Ubah:** tambahkan syarat panjang — `if conf >= self.good_enough and len(plate) >= ALPR_MIN_PLATE_CHARS`.
- **Kenapa:** **67% fragmen motor dan 71% fragmen mobil ber-confidence ≥0,75**, dan
  begitu `strong=True` loop `for roi in self._plate_rois(...)` berhenti sebelum
  `_denoise`/`_sharpen` dan sebelum retry zoom pernah jalan. Efeknya juga menempel
  ke hilir: `worker._maybe_reread` berhenti begitu `vehicle.votes.confidence >= 0.75`,
  jadi satu fragmen palsu memblokir permanen satu-satunya kesempatan kedua motor itu.
- **Selesai kalau:** ada tes yang membuktikan sebuah bacaan 4-karakter ber-confidence
  0,9 **tidak** menghentikan pencarian, sementara bacaan 8-karakter ber-confidence
  0,9 menghentikannya.
- **Risiko:** lebih banyak pass OCR per crop → beban GPU naik. Awasi `detect_fps`
  di dashboard setelah deploy; `OCR_MAX_PASSES` adalah remnya.

### T1.3 — Beri suku cakupan pada scorer pemenang
- [ ] **File:** `backend/app/detection/alpr.py`
- **Anchor:** `_result`, `key=lambda kv: kv[1][0] + 0.12 * (kv[1][1] - 1) - 0.08 * kv[1][2]`
- **Ubah:** tambahkan suku yang menghargai kandidat yang mencakup lebih banyak
  plat. Paling sederhana `+ 0.05 * len(plate)`; lebih benar: bobot dari lebar bbox
  kandidat sebagai fraksi lebar ROI — stempel masa berlaku secara fisik lebih
  sempit daripada nomornya.
- **Catatan implementasi:** bbox-nya **sudah ada** di `_candidates` tetapi dibuang
  sebelum di-return (`for norm, conf, _bbox in candidates:`, dan tipe kembaliannya
  `tuple[str, float, int]`). Versi lebar-bbox berarti meneruskannya lewat
  `_candidates` → `consider` → `votes` → `_result` — sekitar 10 baris, dan ubah
  tipe kembalian `_candidates` sekalian. Versi `len(plate)` tidak butuh itu; kalau
  waktu mepet, kerjakan yang itu dulu dan tandai sisanya sebagai lanjutan.
- **Kenapa:** scorer sekarang tidak punya apa pun yang menghargai cakupan, jadi
  kotak sub-baris bisa mengalahkan baris plat penuh. **Ini regresi bertanggal:**
  fragmen 4–6 karakter pada mobil per tanggal — 08-23 3,8%, 08-24 1,4%, 08-25 2,6%,
  lalu **08-26 18,5%, 08-27 31,7%, 08-28 30,1%, 08-29 26,7%**. Commit `2e08ab9`
  ("Read the plate on a motorcycle capture") bertanggal 08-26 dan itulah yang
  memperkenalkan `_plate_rois` serta memperluas himpunan kandidat. Kualitas mobil
  turun dari 96–98% usable menjadi 68–73% sebagai ongkos menaikkan motor dari ~2%
  ke ~35%.
- **Selesai kalau:** ada tes dengan dua kandidat dari satu ROI — nomor penuh
  ber-confidence sedikit lebih rendah harus mengalahkan stempel pendek
  ber-confidence lebih tinggi.
- **Risiko:** bobotnya perlu disetel. Ambil angkanya dari data ter-review lewat
  T0.1, jangan dikira-kira.

### T1.4 — `_plate_rois` jangan bergantung pada satu deteksi
- [ ] **File:** `backend/app/detection/alpr.py`
- **Anchor:** `_plate_rois`, `if detected is not None: return [detected]`
- **Ubah:** kembalikan `[detected, vehicle_crop]` (dan pita bawah bila crop cukup
  tinggi), bukan `[detected]`.
- **Kenapa:** kotak detektor sekarang adalah **titik kegagalan tunggal** — kalau ia
  mendarat di kaos pengendara, crop yang masih berisi platnya dibuang dan barisnya
  kosong selamanya. Bukti di disk:
  `data/plates/7_1903_20260824_052013_NOPLATE010328.jpg` — "plat" pilihan detektor
  adalah kaos bertuliskan **WE THE FEST**. Bandingkan
  `data/plates/6_589_20260824_032200_NOPLATE366980.jpg` — crop motor utuh yang
  platnya **B 4349 UF-** terbaca jelas oleh mata, barisnya kosong.
- **Selesai kalau:** ada tes yang memastikan ROI kedua ikut dicoba ketika ROI
  pertama tidak menghasilkan bacaan; seluruh tes lama lulus.
- **Risiko:** satu pass OCR ekstra, tapi hanya saat ROI pertama gagal — tangga
  berhenti begitu ada yang terbaca.

### T1.5 — Letterbox crop ke 320×320 sebelum detektor plat
- [ ] **File:** `backend/app/detection/alpr.py`
- **Anchor:** `_detector_roi`, `self._plate_detector.predict(vehicle_crop, ...)`
- **Ubah:** letterbox `vehicle_crop` ke 320×320 konstan (jaga aspek, padding), lalu
  petakan kotak hasil kembali ke koordinat crop asli.
- **Kenapa:** diukur di GPU mesin ini — **40 crop berbeda bentuk = 78,7 ms/panggil;
  40 crop yang sama setelah di-letterbox = 14,6 ms/panggil**; pada crop kendaraan
  4K bahkan 174,6 ms. Ultralytics me-letterbox ke kelipatan 32 sambil menjaga
  aspek, jadi setiap bentuk baru memicu autotune cuDNN ulang. **Ongkos ini dibayar
  hari ini pada setiap pass OCR**, bukan cuma saat scoring — dan itulah sebab utama
  `ALPR_CAPTURE_READ_ATTEMPTS` hanya sanggup jalan saat antrean kosong.
- **Selesai kalau:** ada tes yang memastikan kotak hasil dipetakan balik dengan
  benar (kotak yang diketahui pada gambar sintetis kembali ke koordinat semula
  dalam toleransi 1 px); dan ada pengukuran ms sebelum/sesudah di badan commit.
- **Risiko:** salah petakan balik = semua crop plat bergeser. Wajib bertes.
- **Catatan:** ini gratis 5× dan **mendanai semua gelombang berikutnya**. Kalau
  hanya satu task dari dokumen ini yang dikerjakan, kerjakan yang ini.

---

## Gelombang 2 — pengaman murah & dedupe yang tidak berisiko

Boleh paralel dengan Gelombang 1 (file berbeda).

### T2.1 — Jangan biarkan teks kosong menimpa bacaan yang bagus
- [ ] **File:** `backend/app/detection/worker.py`
- **Anchor:** `_persist_plate`, cabang update — `row.plate_text = text` / `row.confidence = conf`
- **Ubah:** jaga dengan `if text or not row.plate_text` dan `if conf >= row.confidence`.
- **Kenapa:** **bug laten, belum pernah terpicu** (0 baris di DB), tapi jaraknya satu
  class-flip saja. `_do_capture` memanggil `_persist_plate(..., "", 0.0, ..., row_id=vehicle.row_id)`
  dan cabang update menugaskan nilainya tanpa penjaga apa pun. Kendaraan yang sudah
  terbaca sebagai mobil (`row_id` terisi, teks `B1234XYZ` @0,9) yang dilaporkan
  sebagai `motorcycle` untuk satu frame akan lolos `if vehicle.captured` dan
  meng-capture ke barisnya sendiri — file baru bernama `NOPLATE<mikrodetik>` dan
  crop plat lamanya **dihapus**.
- **Selesai kalau:** ada tes yang memanggil `_persist_plate` dengan teks kosong pada
  `row_id` yang sudah punya teks, dan membuktikan teks lama bertahan.
- **Risiko:** rendah. Ini murni penambahan penjaga.

### T2.2 — `_maybe_reread` jangan jalan untuk `person`
- [ ] **File:** `backend/app/detection/worker.py`
- **Anchor:** `submit`, cabang `if capturable and not readable:` → panggilan `_maybe_reread`
- **Ubah:** kelas-gate `_maybe_reread` supaya tidak pernah jalan pada `person`.
- **Kenapa:** `_maybe_reread` tidak pernah memeriksa kelas dan meneruskan
  `det.class_name` langsung ke job, yang sampai ke `_persist_plate` →
  `row.vehicle_class = vehicle_class`. Ini satu-satunya jalur yang bisa membuat
  baris plat berkelas `person`, dan DB punya **tepat 5**: id 2096, 2418, 2269,
  2443, 3007 — masing-masing dengan string plat sungguhan (`S20B` @0,93,
  `B3621TYO` @0,77, `S707TYM` @0,61, `I967Z` @0,96, `Z1689ZZ` @0,89).
- **Selesai kalau:** tes membuktikan deteksi `person` tidak pernah mengantrekan job
  reread. Baris `person` yang sudah telanjur ada dibiarkan — itu data historis.
- **Risiko:** rendah.

### T2.3 — Ledger CaptureGate jangan di-key per kelas
- [ ] **File:** `backend/app/detection/capture_gate.py`
- **Anchor:** `CaptureGate._recent`, `is_repeat`, `record`, `_forget`
- **Ubah:** buang `class_name` dari kunci ledger. Kendaraan yang kelasnya berubah
  tetap satu kendaraan.
- **Kenapa:** **51 klaster (10%) berisi class-flip pada satu objek**, dibuktikan oleh
  teks plat yang identik, bukan oleh dugaan: `B2109UIZ` enam kali dalam 5 detik di
  source 6, bergantian car/truck/car/car/truck/truck. `B1327SLC` 4×,
  `B2302UY(M)` 3×. Kelas adalah tebakan per frame dan tidak pernah divoting.
- **Selesai kalau:** tes di `test_capture_gate.py` membuktikan kotak yang sama di
  bawah dua nama kelas berbeda dikenali sebagai pengulangan. −77 baris.
- **Risiko:** motor dan pejalan kaki yang benar-benar berdampingan bisa saling
  menekan. Pertimbangkan menyisakan `person` di ledger terpisah.

### T2.4 — Gabungkan kotak bersarang di dalam satu frame
- [ ] **File:** `backend/app/detection/worker.py`
- **Anchor:** `submit`, sebelum `self._vehicles.observe(...)`
- **Ubah:** dalam satu frame, kalau dua kotak kelas-capture punya
  `containment >= 0.6` pada kotak yang lebih kecil, perlakukan sebagai satu objek —
  ambil yang lebih besar. `capture_gate.containment()` sudah persis fungsi ini.
- **Kenapa:** **88 baris (3,0%), yang termurah di seluruh taksonomi.** Registry
  secara struktur tidak bisa menanganinya: `_adopt` melewati kendaraan yang
  `last_seen >= now`, dan `submit` mengambil `now` **sekali** per frame — jadi dua
  track id yang hidup bersamaan di satu frame tidak pernah bisa digabung, seberapa
  pun tumpangnya. Terkonfirmasi mata pada 4 dari 4 sampel: 382/383 (kotak A =
  pengendara+motor, kotak B = motor saja, frame sama), 367/368 (containment 1,0),
  1012/1013, 366/367.
- **Selesai kalau:** tes dengan dua deteksi bersarang di satu frame menghasilkan
  satu kandidat capture.
- **Risiko:** dua motor yang benar-benar berdampingan rapat bisa tergabung. Ambang
  0,6 diukur dari pasangan nyata (89 pasang ≥0,6, 23 pasang ≥0,8) — jangan
  diturunkan tanpa mengukur ulang.

### T2.5 — Voting kelas kendaraan
- [ ] **File:** `backend/app/detection/vehicle_registry.py` (+ modul kecil baru, atau perluas `plate_vote.py`)
- **Ubah:** `row.vehicle_class` sekarang last-write-wins dari frame sembarang.
  Jadikan mayoritas berbobot-confidence sepanjang lintasan, persis seperti
  `PlateVoter` untuk teks.
- **Kenapa:** menghapus 5 baris `person` dan flip car/truck dari listing; juga
  menghilangkan kasus `B2213BYI` (car) vs `Z213BYI` (truck) 6 detik terpisah yang
  sebenarnya satu kendaraan.
- **Selesai kalau:** tes membuktikan urutan kelas `car, truck, car, car` menghasilkan
  `car`.
- **Risiko:** rendah. Ikuti gaya `plate_vote.py` — stdlib saja, bisa dites tanpa frame.

---

## Gelombang 3 — jam yang benar & merge identitas

Butuh migrasi. **Jangan mulai sebelum Gelombang 1 dan 2 sudah di-merge dan diukur
ulang dengan T0.1.**

### T3.1 — Simpan waktu frame, bukan waktu tulis
- [ ] **File:** `backend/app/models.py`, `backend/app/database.py`, `backend/app/detection/worker.py`, `backend/app/schemas.py`
- **Ubah:** kolom baru `frame_time DATETIME NULL` pada `plate_reads` (tambahkan ke
  `_ADDED_COLUMNS` dan `_ADDED_INDEXES` di `database.py`). Stempel waktu ambil frame
  ke dalam tuple job saat di-`_offer`, dan simpan. `timestamp` dibiarkan apa adanya
  supaya baris lama tetap berarti.
- **Kenapa:** **`timestamp` adalah waktu baris ditulis, bukan waktu frame.** Lag
  median **2,0 detik**, p95 8 detik, maks 266 detik. Dua baris motor Anda (track
  4392 & 4396) ditulis selisih **49 ms** padahal frame-nya **2 detik** terpisah.
  Setiap jendela dedupe 3s/60s yang ada sekarang mengukur jam yang salah, dan
  jendela 3 detik CaptureGate bisa habis hanya untuk lag antrean.
- **Selesai kalau:** baris baru punya `frame_time` yang lebih awal dari `timestamp`;
  baris lama `NULL` dan tidak ada yang error karenanya; migrasi jalan pada
  `data/app.db` salinan.
- **Risiko:** migrasi pada DB produksi. `database.py` sudah menangani penambahan
  kolom + indeks pada DB yang sudah ada — ikuti polanya, jangan bikin jalur baru.

### T3.2 — Suppression jadi MERGE
- [ ] **File:** `backend/app/detection/capture_gate.py`, `backend/app/detection/worker.py`
- **Ubah:** ledger menyimpan `(box, when, row_id, vid)` alih-alih `(box, when)`.
  `is_repeat` **mengembalikan `row_id`** itu, dan `_persist_plate` dipanggil dengan
  `row_id` tersebut.
- **Kenapa:** ini perubahan tunggal yang melayani goal 1, 3, dan 4 sekaligus.
  `_persist_plate` sudah punya semuanya — pengalamatan `row_id`, update di tempat,
  supersede gambar yang lebih baik. Yang hilang cuma cara menemukan "baris milik
  objek yang tadi ada di sini" ketika `vehicle.row_id` masih None. Penampakan kedua
  jadi **memperbaiki** baris yang ada, bukan menyisipkan baru — dan kasus capture
  kosong (**1755 baris**) tertutup tanpa perlu teks sama sekali.
- **Selesai kalau:** tes mensimulasikan dua penampakan yang tumpang tindih dengan
  vid berbeda dan membuktikan hanya satu baris tertulis.
- **Risiko:** merge yang salah = dua kendaraan jadi satu baris — kebalikan dari
  keluhan sekarang, dan lebih sulit dideteksi. Wajib dibaca bersama T3.4.

### T3.3 — Cek duplikat ulang saat vote bergerak
- [ ] **File:** `backend/app/detection/worker.py`
- **Anchor:** `_update_plate_text`, `_recent_duplicate`
- **Ubah:** `_update_plate_text` harus melakukan yang dilakukan `_persist_plate`:
  cari baris lain di source yang sama dengan teks baru di dalam jendela, dan merge
  ke yang confidence-nya lebih tinggi. Pencocokan pakai **bentuk plat + jarak edit
  ≤2** setelah melipat kelas kebingungan (O/0, I/1/L, Z/2, B/8, S/5, G/6), bukan
  kesamaan byte. `plate_shape()` sudah ada di `plate_vote.py`.
- **Kenapa:** terbukti hitam-putih pada baris **3173**: teks tersimpan `B2213BYI`,
  file-nya `..._I9216B.jpg`. Saat insert vote masih bilang `I9216B`, jadi
  `_recent_duplicate` mencari string itu dan meleset dari baris 3172 (`B2213BYI`,
  19 detik sebelumnya, jelas di dalam jendela 60 detik). **48 baris** sekarang
  menyimpan teks berbeda dari label di nama filenya sendiri. Jarak edit menutup 63
  klaster lagi: `B2213BYI`/`Z213BYI`, `B2955UOS`/`Z955UOS`, `B2302UY`/`B2302UYM`.
- **Selesai kalau:** tes membuktikan dua baris yang berbeda satu karakter dalam
  kelas kebingungan digabung, dan dua plat yang memang berbeda tidak.
- **Risiko:** plat yang memang mirip bisa tergabung. Syarat "bentuk sama" adalah
  pengamannya — jangan dilonggarkan.

### T3.4 — `BLOCKED` Asosiasi prediksi-gerak menggantikan IoU polos
- [ ] **File:** `backend/app/detection/vehicle_registry.py`
- **Anchor:** `_adopt`, `score = iou(v.box, box)`
- **Ubah:** cocokkan kandidat terhadap `predict(seen_box, velocity, now - when)`,
  bukan terhadap kotak basi. Registry sudah menyimpan `box`, `anchor`, `moved_at`,
  `last_seen`; tambahkan kecepatan per kendaraan (delta pusat kotak dan lebar per
  detik, dari `_update`) dan proyeksikan ke depan. Tahan asosiasi ~8 detik, bukan
  `reid_gap=4`.
- **Kenapa:** **penyebab terbesar, ~555 baris (18,8%).** **482 dari 1064 pasang
  duplikat ber-IoU tepat 0,00** — di 4K, motor menempuh lebih dari satu panjang
  kotaknya sendiri antar deteksi. Menurunkan `reid_iou` **tidak bisa** menggantikan
  ini: kotaknya tidak bersentuhan sama sekali. Dalam klaster 3 detik yang berisi
  crop kendaraan utuh, kotak baris kedua lebih kecil dari yang pertama **244 kali
  vs lebih besar 104 kali** — objeknya menjauh, geometri yang justru tidak bisa
  diikuti IoU.
- **Kenapa BLOCKED:** ini task berisiko tertinggi di seluruh rencana. **12% baris
  ter-klaster adalah lalu lintas asli** — 6 dari 32 klaster yang diperiksa mata
  berisi dua kendaraan berbeda sungguhan (444/448: helm+rompi ShopeeFood oranye vs
  pengendara berkepala plontos berbaju gelap, 3 detik terpisah). Terlalu longgar =
  kendaraan asli **hilang** dari log, dan itu kegagalan yang lebih buruk daripada
  duplikat karena tidak kelihatan. Lihat [Keputusan yang menunggu](#keputusan-yang-menunggu).
- **Selesai kalau:** diuji ulang terhadap **407 klaster 3 detik** yang ada, dengan
  angka eksplisit untuk keduanya — berapa duplikat yang hilang, dan berapa kendaraan
  asli yang ikut termakan.
- **Risiko:** tinggi. Kedua kamera menghadap lurus ke gang, jadi alternatif yang
  lebih aman ada: uji koridor-trayektori (y-bawah kotak memetakan monoton ke jarak,
  jadi "objek sama" = kotak bergerak sepanjang koridor dengan perubahan ukuran yang
  konsisten) ditambah cek penampilan murah pada pita pengendara.

---

## Gelombang 4 — frame terbaik

Butuh Gelombang 3 (waktu frame + merge) sudah beres.

### T4.1 — Skor kandidat tiga tingkat + buffer satu kandidat per kendaraan
- [ ] **File:** `backend/app/detection/worker.py`, `backend/app/detection/vehicle_registry.py`
- **Ubah:**
  ```
  Tingkat 0 (gratis, setiap frame deteksi):
      containment zona + lebar kotak >= minimum kelas;
      simpan sebagai kandidat hanya kalau area >= 0,9 x area terbaik sejauh ini
  Tingkat 1 (0,9 ms, hanya yang lolos tingkat 0):
      Laplacian CV_16S pada pita bawah 55% crop kendaraan;
      buang kalau < 0,8 x ketajaman terbaik sejauh ini
  Tingkat 2 (14,6 ms, hanya yang lolos tingkat 1, maks tiap ALPR_ATTEMPT_INTERVAL frame):
      jalankan detektor plat, lalu
      skor = ln(1+lebar_plat) + ln(1+laplacian_plat)
           - 1,2*|aspek - 2,8| + 0,2*ln(1+area_kendaraan)
      veto: tidak ada kotak plat, atau lebar_plat < 130 px  ->  bukan kandidat
  ```
  Buffer di `Vehicle` (yang sudah selamat dari renumbering): `best_score`,
  `best_crop`, `best_plate`, `best_box`, `best_evidence` — evidence frame
  **diperkecil ke 960×540**, bukan 4K. Ganti kelimanya sekaligus saat skor membaik.
- **Kenapa:** pemilihan sekarang salah dua kali. `_maybe_capture` menembak pada
  **frame pertama** yang 99% masuk zona dan tidak pernah menengok lagi — untuk
  kendaraan mendekat itu justru frame terkecil dan terjauh; dalam satu lintasan
  kotak terlebar **median 1,92×** yang tersempit, >2× pada 47%. Lalu `_record_read`
  memilih gambar lewat `better_image = conf > vehicle.best_conf`, padahal confidence
  **tidak terkalibrasi** (band 0,6–0,8 → 0% exact, n=31) dan 67% fragmen skor ≥0,75
  — jadi sistem **aktif memilih frame fragmen sebagai bukti terbaik**.
  Sinyal yang dipakai diukur (AUC, 0,5 = tak berguna): aspek kotak plat **0,801**,
  lebar kotak plat 0,737, luas kotak kendaraan 0,722, Laplacian crop plat 0,702 —
  dan **confidence detektor plat 0,586, terbalik di atas 0,8, jangan dipakai**.
  Lantai keras: **tidak ada satu pun bacaan tepat di bawah ~130 px lebar kotak plat**
  (0 exact dari 30 baris).
- **Memori:** ~**2,6 MB per kendaraan** (crop 0,93 + crop plat 0,03 + evidence
  960×540 1,6), ~13 MB per source pada 5 kendaraan bersamaan, ~26 MB untuk dua
  source. Ini **lebih hemat dari sekarang**: `submit` dan `_maybe_capture` menyalin
  frame 4K penuh (24,9 MB) per attempt, antrean 8 dalam, dan satu mobil pada
  `ALPR_MAX_ATTEMPTS=24`/interval 2 bisa memicu 12 salinan.
- **Selesai kalau:** pada pasangan `7_4392` vs `7_4396`, skornya memilih yang
  pertama (kotak 231×342 area 79002 vs 153×275 area 42075 — 1,88× lebih banyak
  piksel). `ln(area)+ln(lap)` memberi 17,82 vs 17,59; verifikasi rumus penuh juga.
- **Risiko:** bobotnya diturunkan dari AUC dan uji-pilih 19 klaster — **titik awal,
  bukan hasil fitting.** Turunkan ulang lewat `scripts/export_plate_dataset.py`
  setelah beberapa ratus baris ditinjau. Target aspek 2,8 adalah median bacaan
  tepat; plat mobil Indonesia 395×135 = 2,93 dan motor 250×105 = 2,38, jadi kalau
  mau per kelas pakai 2,9 dan 2,4.

### T4.2 — Aturan commit
- [ ] **File:** `backend/app/detection/worker.py`
- **Ubah:** commit pada yang **paling awal** dari: (a) puncak terlewat — area kotak
  turun 4 frame deteksi beruntun **dan** skor tidak membaik selama 4 frame;
  (b) keluar zona — containment turun di bawah `ALPR_ZONE_MIN_OVERLAP`; (c) track
  hilang — tidak terlihat selama `ALPR_REID_GAP_SECONDS`; (d) timeout keras 4 detik
  di dalam zona, atau `attempts >= ALPR_MAX_ATTEMPTS`.
- **Kenapa:** kendaraan menjauh tidak boleh menunggu exit zona yang dicapainya
  perlahan. Arah maju/mundur **gratis**: `vehicle_registry._update` sudah menghitung
  `growth = abs((x2-x1) - aw)` terhadap anchor — simpan tandanya, jangan
  absolutkan, dan arahnya sudah didapat. Klaster satu source membentang median
  1,29 detik, maks 8,37 — 4 detik menutup median dan membatasi ekornya;
  `ALPR_STATIONARY_SECONDS=20` jauh terlalu panjang untuk jadi batas di sini.
- **Latensi yang ditambahkan:** kendaraan menjauh +0,16–0,4 detik (4 frame pada
  10–25 detect fps) — **lebih kecil dari lag antrean yang sudah ada sekarang**.
  Kendaraan mendekat: nol, karena puncaknya memang frame exit.
- **Selesai kalau:** tes untuk keempat pemicu, termasuk kendaraan yang berhenti di
  dalam zona (harus kena timeout (d), bukan menggantung).

### T4.3 — Tetap tulis baris di frame pertama, perbaiki di tempat
- [ ] **File:** `backend/app/detection/worker.py`
- **Ubah:** tulis baris pada frame tingkat-0 pertama persis seperti `_maybe_capture`
  sekarang, lalu setiap perbaikan skor memanggil `_persist_plate` dengan `row_id`
  yang **sama**. Jalur itu juga harus mengirim snapshot, kalau tidak evidence frame
  membeku di frame pertama selamanya.
- **Kenapa:** **jangan pakai tunda-tulis murni.** Buffer itu memori proses; kalau
  worker mati di tengah lintasan, lintasannya hilang diam-diam — pada puncak 9
  baris/menit/source itu 1–2 baris per crash. Mekanisme `row_id` yang sudah ada
  sudah menangani ini: `_persist_plate` meng-update di tempat dan hanya menghapus
  file yang benar-benar digantikan.
- **Ongkos:** 3–5 pasang tulis JPEG per lintasan, bukan 1. `_discard_plate_image`
  sudah membersihkan di belakangnya.
- **Selesai kalau:** mematikan worker di tengah lintasan tetap meninggalkan satu
  baris dengan gambar yang valid.

---

## Gelombang 5 — plafon motor & malam

### T5.1 — `BLOCKED oleh T0.2` Perbaiki lokalisasi plat motor
- [ ] **File:** `backend/app/detection/alpr.py`, `backend/app/config.py`
- **Ubah:** naikkan `PLATE_IMGSZ` ke 640 khusus jalur capture, **atau** perbesar
  crop kendaraan 2–3× sebelum `_detector_roi`. T0.2 yang memutuskan mana.
- **Kenapa:** **ini plafon motor yang sebenarnya — bukan OCR.** Dari 1755 baris
  motor kosong, **1625 (92,6%) tidak punya plat yang ditemukan sama sekali**;
  yang tersimpan crop motor utuh (median 225×350, aspek 0,64). Bandingkan mobil:
  93,5% crop-nya berbentuk plat (median 214×76, aspek 2,87). Detektor menerima crop
  motor 225 px pada `PLATE_IMGSZ=320`, jadi plat ~40×20 px mendarat jadi ~35×18 px.
  Platnya ada di gambar — terbaca mata dari baris-baris kosong itu: `B 6465 UTY`,
  `B 3112 PLC`, `B 3117 UCG`, `B 3064 YAU`, `T 4096 VP`, `B 3103 LUP`.
- **Hasil T0.2:** _(isi di sini)_
- **Risiko:** 640 menggandakan ongkos detektor. T1.5 (letterbox, 5×) harus sudah
  masuk lebih dulu supaya ada anggarannya.

### T5.2 — Beri motor anggaran OCR sungguhan
- [ ] **File:** `backend/app/detection/worker.py`
- **Anchor:** `_maybe_reread` dan `_do_capture`, keduanya menguji `self._queue.empty()`
- **Ubah:** ganti tes antrean-kosong dengan antrean dua-prioritas: capture
  berprioritas rendah tapi **tidak kelaparan**. Atau pindahkan `motorcycle` ke
  `ALPR_CLASSES`.
- **Kenapa:** motor adalah **72% lalu lintas** dan hanya kebagian sisa antrean.
  Lebih dari itu, **tesnya salah arah**: `_do_capture` menjalankan `read_plate`
  **inline di thread `alpr` yang sama** yang menguras antrean, jadi `empty()`
  bernilai True justru saat thread sedang membaca, dan False saat ia cuma menumpuk.
  Kontensinya juga bukan mobil-vs-motor: hanya 0,6–1,9% baris motor kosong
  berbarengan dengan baris mobil/truk dalam 5 detik, tapi **49,3%** berbarengan
  dengan baris lain dalam 3 detik (vs 33,7% untuk yang berhasil terbaca).
- **Selesai kalau:** tes membuktikan capture tetap dilayani ketika antrean sibuk
  terus-menerus.
- **Risiko:** mengurangi anggaran mobil. Ukur mobil sebelum/sesudah dengan T0.1 —
  jangan sampai memperbaiki motor sambil merusak mobil, yang persis kesalahan yang
  dibuat commit `2e08ab9`.

### T5.3 — Malam hari
- [ ] **File:** `backend/app/detection/alpr.py`
- **Ubah:** tambahkan varian gamma/pengurang eksposur ke tangga preprocessing.
- **Kenapa:** motor **1,6% usable** di malam hari (4 plat dari 245 lintasan), mobil
  40,7% (siang 77,9%). Keempat operator yang ada (`_equalize` CLAHE, `_denoise`
  bilateral, `_smooth`, `_sharpen`) semuanya penguat kontras **lokal** — tak satu
  pun bisa menyelamatkan plat retroreflektif yang gosong kena IR.
- **Risiko / catatan jujur:** menyetel shutter dan gain **di kamera** kemungkinan
  besar mengalahkan apa pun yang dikerjakan di software. Kerjakan ini terakhir, dan
  coba sisi kamera dulu.

---

## Keputusan yang menunggu

1. **T1.1 — lantai 7 karakter.** Setelah ini, baris yang tadinya menampilkan `C8JC`
   akan kosong. Listing akan **terlihat** lebih buruk padahal lebih jujur — bacaan
   yang dibuang itu 0 dari 38 benar. Setuju? Ada plat 1–2 digit (kendaraan dinas)
   di lokasi ini yang perlu ditampung?
2. **T3.4 — merge identitas.** Ini menghapus ~555 baris duplikat, tapi ~12% baris
   ter-klaster adalah kendaraan asli. Mana yang lebih buruk untuk Anda: **satu
   kendaraan tercatat dua kali**, atau **satu kendaraan tidak tercatat sama sekali**?
   Jawabannya menentukan seberapa agresif ambangnya disetel.
3. **Goal 2 untuk jalur read.** Jalur capture sudah menuntut 99% masuk zona dan itu
   sudah terpenuhi — hanya 8,1% baris motor di bawahnya sejak 08-26. Tapi jalur
   **read** memakai `_in_zone >= 0.5`, jadi mobil bisa dibaca saat separuh badan di
   zona. Itu disengaja (read mau frame paling awal yang terbaca). Mau diseragamkan
   ke 99%, dengan konsekuensi kehilangan sebagian bacaan mobil?
4. **Urutan.** Saya sarankan Gelombang 1 dulu (satu hari, membelah CER semua kelas
   dan membalik regresi mobil), **lalu ukur ulang dengan batch review baru** —
   karena 138 review yang ada dikumpulkan di bawah scorer yang sedang regresi. Baru
   Gelombang 2. Setuju, atau ada yang lebih mendesak?

---

## Yang sengaja TIDAK dikerjakan

- **`ALPR_MIN_VEHICLE_WIDTH=160`** — diukur terhadap lebar kotak nyata, tidak
  mengikat di kedua kamera. Motor melewatinya lewat `alpr_capture_min_width=48`
  (hanya 14,3% kotak motor di bawah 160), mobil jauh lebih lebar (median 953 px).
  Menaikkannya hanya mulai membuang mobil; menurunkannya memasukkan kendaraan yang
  platnya sudah di bawah 40 px.
- **Ambang zona `0.99`** — goal 2 sudah terpenuhi. Duplikat yang tersisa adalah
  kegagalan identitas, bukan kegagalan zona.
- **Confidence detektor plat sebagai skor frame** — AUC 0,586 dan **terbalik** di
  atas 0,8 (band 0,60–0,80 → jarak edit rata-rata 2,97 dengan 13 exact dari 75;
  band 0,80–1,01 → 4,06 dengan 1 exact dari 16).
- **Jarak dari tepi frame, tinggi crop plat sendirian, densitas tepi, kontras crop**
  — AUC 0,497 / 0,500 / 0,476 / 0,578. Semuanya dalam derau. Terdengar masuk akal,
  terukur tidak berguna.
- **Menyetel truk secara terpisah** — n=64 terlalu kecil untuk menyimpulkan apa pun
  di luar tingkat fragmen 36% yang sama dengan kelas lain.

---

## Catatan pengawasan

**Voting antar-frame yang baru masuk di `7b66c13` perlu diawasi.** Dari 108 baris
ter-review yang legible, hanya 2 yang teksnya berubah karena voting — **satu
memburuk** (`B2025TRY` yang benar menjadi `BB2025TRY`), satu tetap salah
(`Z955US` → `Z955UOS`, kebenaran `2955UOS`). Sampelnya terlalu kecil untuk
menyimpulkan apa pun, tapi arahnya bukan yang diharapkan. Periksa lagi setelah
T0.1 dan setelah beberapa ratus baris baru ditinjau; kalau polanya bertahan,
`ALPR_READ_ALL_ATTEMPTS=false` adalah tombol untuk mengembalikannya.
