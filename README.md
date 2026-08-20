# Detection Dashboard

Sistem **object detection berbasis website** untuk menghitung dan menganalisis
lalu lintas (mobil, orang, truk, motor, bus) dari berbagai sumber video, dengan
**line counting** dan **pembacaan plat nomor (ANPR)**.

## Fitur

- 🎯 **Deteksi objek** (YOLO11): mobil, orang, motor, truk, bus — user memilih
  kelas mana yang aktif per source.
- 📥 **Banyak jenis input**: YouTube, RTSP (CCTV), RTMP, HLS, HTTP/MJPEG, file.
- 📊 **Dashboard**: tambah source, lihat berapa source aktif, thumbnail live tiap
  source, status real-time.
- 📏 **Line counting** (fitur inti): gambar satu garis hitung dengan mengklik dua
  titik pada snapshot video. Objek yang **melintasi** garis dihitung terpisah per
  kelas dan per arah. Logika berbasis perpotongan segmen + track-ID sehingga
  akurat dan tidak double-count (lihat `backend/app/detection/line_counter.py`).
- 🚗 **ANPR / Plat nomor**: membaca plat kendaraan (format Indonesia) via
  EasyOCR, menyimpan teks + potongan gambar ke database lokal.
- 🧑 **Pengenalan wajah (face recognition)**: daftarkan orang (nama + foto),
  sistem mengenali wajah pada stream (label nama) dan mencatat kemunculan
  (sighting) ke DB. Ringan & OpenCV-native (**YuNet** deteksi + **SFace**
  embedding), toggle per source. Wajah asing = "unknown".
- 🗄️ **Penyimpanan lokal (SQLite)** dengan **retensi otomatis 7 hari**.
- 🔐 **Login** (satu admin, JWT).
- 🐳 **Docker** untuk distribusi mudah (CPU & GPU).

## Arsitektur

```
Browser (React SPA)
   │  REST + MJPEG
   ▼
FastAPI (satu proses)  ── DetectionManager ── worker proses per source
   │                                              │
   ├─ SQLite (app.db, WAL)  ◄─────────────────────┤ CountEvent / PlateRead
   ├─ data/plates/*.jpg     ◄─────────────────────┤ crop plat
   └─ APScheduler (retensi 7 hari)
```

Setiap source berjalan di **proses terpisah** (mendukung 4–10 stream). Worker:
baca frame → YOLO deteksi + ByteTrack → line counting → ANPR → simpan ke DB →
publish frame beranotasi (JPEG) ke shared state untuk streaming MJPEG.

## Menjalankan dengan Docker (disarankan)

Prasyarat: Docker + Docker Compose.

```bash
cp .env.example .env       # ubah ADMIN_PASSWORD & JWT_SECRET
docker compose up --build
```

Buka http://localhost:8000 dan login (default `admin` / password dari `.env`).

Data (DB + gambar plat) tersimpan di folder `./data` (volume), jadi tetap ada
walau container di-restart atau dipindah ke komputer lain.

### Dengan GPU NVIDIA

Butuh **NVIDIA Container Toolkit** di host.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Image GPU memakai torch CUDA dan `DEVICE=cuda`.

## Menjalankan di Mac mini M4 / M4 Pro (Apple Silicon) 🍎

Docker Desktop di macOS **tidak bisa** meneruskan GPU / Neural Engine ke dalam
container, jadi untuk memakai akselerasi hardware jalankan secara **native**:

```bash
./scripts/setup_macos.sh     # venv + dependensi + model + build frontend
# edit .env (ADMIN_PASSWORD, JWT_SECRET)
./scripts/run_macos.sh       # http://localhost:8000
```

`setup_macos.sh` membuat `backend/.venv`, memasang PyTorch build Apple Silicon
(dengan Metal/MPS), mengunduh bobot model ke `data/weights/`, dan menyalin
`.env.macos.example` → `.env` bila belum ada.

Cek akselerasi benar-benar aktif:

```bash
curl -s localhost:8000/api/health | python3 -m json.tool
# "device": "mps", "ffmpeg_hwaccel": "videotoolbox", "cpu": "Apple M4 Pro"
```

### Hasil pengukuran (Mac mini M4 Pro, 12-core: 8P + 4E)

Deteksi saja, input 1080p, `imgsz=640` — diukur dengan `scripts/benchmark.py`:

| Model | Perangkat | FPS | ms/frame |
|---|---|---:|---:|
| yolo11n | CPU | 40 | 25.2 |
| yolo11n | **GPU (mps)** | **108** | 9.3 |
| yolo11n | Neural Engine (CoreML) | 113 | 8.9 |
| yolo11s | GPU (mps) | 91 | 11.0 |

Artinya M4 Pro sanggup memakai **`yolo11s`** (lebih akurat dari `yolo11n`) dan
tetap punya sisa tenaga. Dengan `FRAME_STRIDE=2`, 4 stream 1080p bersamaan
memakai ±64% dari satu core per worker (±570 MB RAM per stream) — sisa banyak
untuk 8–10 stream.

Ukur sendiri sebelum memutuskan:

```bash
backend/.venv/bin/python scripts/benchmark.py --model data/weights/yolo11s.pt --half
```

### Apa saja yang dioptimalkan

- **GPU Metal (MPS)** — `DEVICE=mps` (atau kosong = auto `cuda → mps → cpu`).
- **Neural Engine (CoreML)** — opsional, paling hemat daya dan membebaskan GPU:
  ```bash
  backend/.venv/bin/pip install coremltools
  backend/.venv/bin/python scripts/export_coreml.py --model data/weights/yolo11s.pt
  # lalu set YOLO_MODEL=data/weights/yolo11s.mlpackage (INFERENCE_IMGSZ harus sama)
  ```
- **Decoding video di hardware** — `FFMPEG_HWACCEL=videotoolbox` memindahkan
  decoding H.264/HEVC ke media engine, jadi core CPU bebas untuk inference.
  Otomatis balik ke software bila suatu stream menolak.
- **Budget thread per worker** — tiap source jalan di proses sendiri; tanpa
  batas, OpenCV dan torch sama-sama mengambil semua core dan saling rebutan.
  `THREADS_PER_WORKER=0` membagi P-core dengan `EXPECTED_STREAMS`.
- **`PROCESS_WIDTH`** — deteksi, anotasi dan streaming memakai frame yang
  sudah diperkecil (mis. 1280), sementara **ANPR tetap memotong dari frame
  resolusi penuh** supaya plat tetap terbaca.
- **Warmup model** saat worker start, supaya kompilasi kernel Metal / load
  CoreML tidak menahan frame pertama.
- **RTSP over TCP** (`RTSP_TRANSPORT_TCP=true`) — jauh lebih sedikit frame rusak
  dari CCTV dibanding UDP.
- **fp16** (`INFERENCE_HALF`) tersedia tapi **default mati**: pada M-series
  hasilnya campur (yolo11n sedikit lebih cepat, yolo11s justru lebih lambat).
- **Tampilan lepas dari deteksi** — frame di antara dua deteksi digambar ulang
  dengan kotak terakhir, jadi video tetap mulus meski deteksi jalan 15 fps.
- **Buffer per jenis protokol** — RTSP/RTMP dibuang frame lamanya (biar tetap
  live), HLS/YouTube di-buffer lalu dilepas sesuai fps aslinya (biar burst per
  segmen tidak terlihat patah-patah).
- **Backpressure pada decoder** — thread decoder yang lari bebas membuat loop
  deteksi kelaparan GIL; menahannya menaikkan throughput **12.5 → 30 fps**.
- **Streaming hanya mengirim frame baru** — endpoint MJPEG membandingkan nomor
  urut frame dulu, jadi tidak ada JPEG yang dikirim ulang percuma.

Masih bisa memakai Docker di Mac (`docker compose up --build`) — jalan normal,
tetapi **CPU-only**, jadi ±2.7× lebih lambat dari mode native.

## Menjalankan di Windows (AMD Ryzen / Intel Core) 🪟

Docker Desktop di Windows tidak bisa mem-passthrough iGPU, jadi instalasi
native adalah satu-satunya cara mendapat akselerasi di dua mesin ini. Jalur
inferensinya **OpenVINO** untuk keduanya:

| Mesin | Device | Model | Alasan |
| --- | --- | --- | --- |
| Ryzen 7 PRO 7840U (8C/16T, Zen 4) | `intel:cpu` | `yolo11s` INT8 | Plugin CPU OpenVINO vendor-neutral dan memakai AVX-512/VNNI Zen 4 |
| Core i7 gen-7 (2C/4T, HD Graphics) | `intel:gpu` | `yolo11n` FP16, imgsz 480 | Dua core CPU terlalu sedikit; iGPU Gen9 didukung plugin GPU OpenVINO |

```powershell
git clone <repo> && cd detection
.\scripts\setup_windows.ps1        # deteksi CPU -> pilih profil, export model
# edit .env (ADMIN_PASSWORD, JWT_SECRET)
.\scripts\run_windows.ps1          # http://localhost:8000
```

Script setup memilih profil dari vendor CPU, memasang torch CPU + OpenVINO,
mengunduh bobot, meng-export IR OpenVINO, dan menyalin `.env.amd.example` atau
`.env.intel.example` menjadi `.env`. Paksa profil dengan
`.\scripts\setup_windows.ps1 -Hardware intel`.

Cek device yang benar-benar dipakai:

```powershell
curl http://localhost:8000/api/health
# "device": "intel:cpu", "backend": "openvino",
# "physical_cores": 8, "openvino_devices": ["CPU"]
```

### Hasil pengukuran (Ryzen 7 PRO 7840U, 1080p, imgsz=640, deteksi saja)

| Model | torch CPU | OpenVINO CPU | OpenVINO CPU INT8 |
| --- | --- | --- | --- |
| yolo11n | 33 fps | 60 fps | **73 fps** |
| yolo11s | 17 fps | — | **44 fps** |

Dipin ke 2 core (jatah 1 worker saat `EXPECTED_STREAMS=4`), yolo11n INT8 masih
55 fps — cukup untuk empat stream 25 fps dengan `FRAME_STRIDE=2`.

Ukur di mesin sendiri:

```powershell
backend\.venv\Scripts\python scripts\benchmark.py `
  --model data\weights\yolo11s_int8_openvino_model --imgsz 640
```

### Apa saja yang dioptimalkan

- **Inferensi lewat OpenVINO** — export sekali dengan
  `scripts/export_openvino.py` (`--half` untuk iGPU Intel, `--int8` untuk Zen 4),
  lalu `YOLO_MODEL` diarahkan ke direktori `*_openvino_model`. Device dipilih
  otomatis: iGPU Intel bila ada, kalau tidak plugin CPU.
  ⚠️ IR punya **ukuran input tetap** — `INFERENCE_IMGSZ` wajib sama dengan
  `--imgsz` saat export.
- **Hitung core fisik, bukan logis** — sebelumnya `os.cpu_count()` melaporkan 16
  di 7840U, sehingga setiap worker minta thread 2× lebih banyak dari core yang
  ada dan saling rebut SMT sibling. Sekarang 8.
- **Affinity per worker** — tiap worker dikunci ke potongan core-nya sendiri.
  Ini satu-satunya tuas yang mengikat plugin CPU OpenVINO: ia menjadwal di atas
  TBB dan mengabaikan `OMP_NUM_THREADS`, tetapi menghormati affinity mask.
  Matikan dengan `WORKER_CPU_AFFINITY=off`.
- **Budget thread sadar-backend** — worker OpenVINO **CPU** mendapat jatah
  thread penuh, worker akselerator (MPS/CUDA/CoreML/OpenVINO GPU) tetap 2.
- **Decode video di hardware** — `FFMPEG_HWACCEL=auto` lewat
  `CAP_PROP_HW_ACCELERATION` milik OpenCV: D3D11VA di Windows, VAAPI di Linux,
  VideoToolbox di macOS, dengan fallback software di dalam OpenCV sendiri.
  Bisa dipaksa: `d3d11va`, `qsv`, `vaapi`, atau `off`.

### Kenapa iGPU Radeon 780M tidak dipakai

Plugin GPU OpenVINO hanya mendukung GPU Intel (`Core().available_devices` di
7840U memang hanya melaporkan `CPU`). Jalur DirectML butuh backend ONNX kustom
karena ultralytics tidak pernah mendaftarkan `DmlExecutionProvider`. Karena
plugin CPU dengan INT8 sudah memberi 2.2×, kompleksitas itu tidak sepadan.

## Menjalankan tanpa Docker (development)

**Backend:**
```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend (dev server dengan proxy ke :8000):**
```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

Untuk produksi single-origin: `npm run build` lalu jalankan backend — FastAPI
otomatis menyajikan `frontend/dist`.

## Cara pakai

1. **Login**.
2. **+ Tambah Source** → pilih tipe (mis. YouTube), tempel URL, centang objek
   (mis. Mobil, Truk, Motor), aktifkan ANPR bila perlu → Simpan.
3. Klik **Start** pada kartu source. Thumbnail live muncul.
4. Buka **Detail** → tab **Atur Garis Hitung** → klik dua titik untuk menggambar
   garis melintang jalan, beri label arah (mis. `masuk`/`keluar`) → **Simpan
   Garis**. Worker otomatis restart dengan garis baru.
5. Kembali ke tab **Live** — hitungan per kelas/arah bertambah saat kendaraan
   melintasi garis. Plat nomor terbaca muncul di panel & halaman **Plat Nomor**.

## Konfigurasi (env)

| Variabel | Default | Keterangan |
|---|---|---|
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | admin / admin | kredensial login |
| `JWT_SECRET` | change-me | rahasia token — wajib diganti |
| `DEVICE` | *(auto)* | `cpu` / `cuda` |
| `YOLO_MODEL` | yolo11n.pt | ganti ke `yolo11s.pt` dll untuk akurasi lebih |
| `FRAME_STRIDE` | 2 | proses 1 dari N frame (hemat CPU) |
| `INFERENCE_IMGSZ` | 640 | resolusi inference |
| `INFERENCE_HALF` | false | inference fp16 (khusus GPU, ukur dulu) |
| `PROCESS_WIDTH` | 0 | perkecil frame ke lebar ini sebelum deteksi (0 = asli) |
| `THREADS_PER_WORKER` | 0 | thread CPU per worker (0 = otomatis dari jumlah P-core) |
| `EXPECTED_STREAMS` | 4 | perkiraan jumlah source aktif, untuk pembagian thread |
| `FFMPEG_HWACCEL` | *(kosong)* | `videotoolbox` (macOS) / `cuda` (NVIDIA) |
| `RTSP_TRANSPORT_TCP` | true | paksa RTSP lewat TCP |
| `YOUTUBE_MAX_HEIGHT` | 720 | batas resolusi stream YouTube |
| `CAPTURE_BUFFER_SECONDS` | 2 | jitter buffer untuk HLS/YouTube (RAM: ±83 MB/detik pada 720p) |
| `MJPEG_FPS` | 15 | frame per detik yang dikirim ke browser |
| `JPEG_QUALITY` | 70 | kualitas JPEG stream |
| `OCR_DEVICE` | *(auto)* | perangkat EasyOCR; MPS sengaja tetap di CPU |
| `CONF_THRESHOLD` | 0.35 | ambang confidence deteksi |
| `ALPR_ENABLED` | true | aktif/nonaktif ANPR global |
| `FACE_ENABLED` | true | aktif/nonaktif face recognition global |
| `FACE_SIMILARITY_THRESHOLD` | 0.363 | ambang cosine SFace (lebih tinggi = lebih ketat) |
| `RETENTION_DAYS` | 7 | umur data sebelum dihapus |

## Testing logika line counting

```bash
cd backend
PYTHONPATH=. python tests/test_line_counter.py
PYTHONPATH=. python tests/test_runtime_tuning.py   # helper tuning performa
```

## Catatan performa & akurasi

- **CPU**: gunakan `yolo11n` + `FRAME_STRIDE` lebih besar untuk 4–10 stream.
- **GPU**: bisa model lebih besar & lebih banyak stream.
- **Apple Silicon**: lihat bagian *Menjalankan di Mac mini M4 / M4 Pro* — jalankan
  native (bukan Docker) supaya GPU/Neural Engine terpakai.
- Urutan yang paling berpengaruh saat kurang kencang: naikkan `FRAME_STRIDE` →
  set `PROCESS_WIDTH` → turunkan `INFERENCE_IMGSZ` → model lebih kecil.
- **Akurasi ANPR** bergantung resolusi/sudut/pencahayaan CCTV; hasil disimpan
  beserta nilai confidence. ANPR bisa dimatikan per source untuk menghemat CPU.
- **Face recognition (YuNet + SFace)** ringan tapi akurasinya bukan kelas
  InsightFace/ArcFace; paling baik untuk wajah frontal & jelas. Enrol beberapa
  foto per orang untuk hasil lebih stabil. Cocokkan `FACE_SIMILARITY_THRESHOLD`
  bila terlalu banyak false match (naikkan) atau sering "unknown" (turunkan).

## Pengenalan Wajah (Face Recognition)

1. Buka menu **Wajah** → **Daftarkan Orang**: isi nama + unggah foto wajah
   frontal yang jelas. (Bisa daftarkan beberapa foto untuk orang yang sama.)
2. Saat **menambah source**, centang **"Pengenalan wajah (face recognition)"**.
   Aktifkan juga kelas **orang** bila ingin deteksi objek orang sekaligus.
3. Start source → wajah yang dikenal diberi **label nama** pada video; kemunculan
   dicatat di tabel **Kemunculan Terdeteksi** (dengan crop + similarity).
   Wajah tak dikenal diberi label **"unknown"** (default tidak disimpan).
4. Enrolment baru otomatis dipakai worker dalam ~30 detik tanpa restart.

> ⚠️ Data biometrik bersifat sensitif — gunakan hanya untuk keperluan yang sah
> dan sesuai regulasi setempat.

## Troubleshooting

### Video live terputus-putus / tidak lancar

Buka **Detail** source — di bawah video ada angka fps. Itu menunjukkan di mana
letak masalahnya:

| Yang terlihat | Artinya | Tindakan |
|---|---|---|
| `masuk` rendah (< 10) | **sumbernya** yang lambat — kamera/jaringan/YouTube | bukan masalah server; cek jaringan atau ganti sumber |
| `masuk` normal, `deteksi` rendah | inference tidak terkejar | naikkan `FRAME_STRIDE`, set `PROCESS_WIDTH`, atau pakai model lebih kecil |
| `dibuang` tinggi | decoding lebih cepat dari pemrosesan | naikkan `FRAME_STRIDE` |
| `tampil` jauh di bawah `masuk` | batas `MJPEG_FPS` | naikkan `MJPEG_FPS` (ideal ≥ fps kamera) |

Lewat API: `curl -s localhost:8000/sources -H "authorization: Bearer <token>"`.

Beberapa hal yang perlu diketahui:

- **`MJPEG_FPS` sebaiknya ≥ fps kamera.** Frame hanya datang pada irama kamera,
  jadi kamera 30 fps dengan `MJPEG_FPS=20` akan jatuh ke 15 fps. Isi 30 untuk
  video mulus, atau turunkan ke 10–15 untuk menghemat CPU saat banyak tile.
- **Kemulusan video tidak lagi terikat kecepatan deteksi.** Frame di antara dua
  deteksi digambar ulang memakai kotak terakhir, jadi `FRAME_STRIDE=2` tidak
  membuat video jadi setengah lambat.
- **YouTube live sering tersendat dari sananya.** Diukur pada stream CCTV live
  (hanya tersedia HLS): tanpa deteksi sama sekali, hanya decoding, tetap ada
  11–12 jeda >0.5 detik per 75 detik dengan jeda terpanjang **14 detik** — sama
  saja di 360p maupun 720p. Itu batas sumbernya, bukan Mac-nya. **RTSP langsung
  dari CCTV jauh lebih stabil** dan itu jalur yang dioptimalkan di sini.
- **`CAPTURE_BUFFER_SECONDS`** memperhalus HLS yang datang per segmen, tapi
  menyimpan frame mentah: ±83 MB per detik buffer pada 720p. Menaikkannya di
  atas ±4 detik jarang sepadan.

### YouTube: `No video formats found!` / source status `error`

Karena YouTube memang rentan berubah, jika suatu saat error `No video formats
found` muncul lagi, cukup **rebuild** agar `yt-dlp` ter-update:

```bash
docker compose up --build
```

Atau cepatnya, update `yt-dlp` di dalam container yang sedang jalan lalu restart
source-nya (Stop lalu Start dari dashboard):

```bash
docker compose exec app pip install -U yt-dlp
```

Untuk video yang tetap gagal (privat, age/geo-restricted, members-only), itu
**batasan konten, bukan bug**. Alasan kegagalan juga tampil pada status source
di dashboard.
