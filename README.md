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
  kelas dan per arah. Diuji terhadap titik **kontak roda** kendaraan (bukan titik
  tengah box) dan memakai pita histeresis, sehingga box yang bergoyang tidak
  menghasilkan hitungan ganda (lihat `backend/app/detection/line_counter.py`).
- 🚗 **ANPR / Plat nomor**: membaca plat kendaraan (format Indonesia) via
  EasyOCR, menyimpan teks + potongan gambar ke database lokal. Anda menggambar
  **zona baca** — kotak di bagian gambar tempat plat benar-benar terbaca — dan
  program hanya mencoba di situ. Setiap pembacaan juga menyimpan **satu frame
  penuh** dengan kendaraannya dikotaki, supaya bisa dipastikan plat itu menempel
  pada kendaraan yang mana.
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

Image GPU memakai torch CUDA (cu130, di atas base CUDA 13) dan `DEVICE=cuda`.
CUDA 13 dipilih karena RTX seri 50 adalah sm_120 (Blackwell) dan wheel cu128 ke
bawah tidak punya kernel untuknya; kartu lama tetap didukung.

Di Windows, instalasi native lewat `.\scripts\setup_windows.ps1` umumnya lebih
praktis daripada Docker — lihat
[Menjalankan di Windows dengan GPU NVIDIA](#menjalankan-di-windows-dengan-gpu-nvidia-rtx-).

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
- **Tampilan jalan di thread sendiri** — preview mengambil frame terbaru
  langsung dari decoder dan menggambar ulang kotak terakhir pada `MJPEG_FPS`,
  jadi video tetap mulus berapa pun lambatnya deteksi.
- **Buffer per jenis protokol** — RTSP/RTMP dibuang frame lamanya (biar tetap
  live), HLS/YouTube di-buffer lalu dilepas sesuai fps aslinya (biar burst per
  segmen tidak terlihat patah-patah).
- **Backpressure pada decoder** — thread decoder yang lari bebas membuat loop
  deteksi kelaparan GIL; menahannya menaikkan throughput **12.5 → 30 fps**.
- **Streaming hanya mengirim frame baru** — endpoint MJPEG membandingkan nomor
  urut frame dulu, jadi tidak ada JPEG yang dikirim ulang percuma.

Masih bisa memakai Docker di Mac (`docker compose up --build`) — jalan normal,
tetapi **CPU-only**, jadi ±2.7× lebih lambat dari mode native.

## Menjalankan di Windows dengan GPU NVIDIA (RTX) 🟩

Jalur tercepat di Windows. Deteksi kendaraan, OCR plat (EasyOCR) dan detektor
plat semuanya jalan di GPU; CPU hanya kebagian decode, resize, gambar box dan
encode JPEG.

```powershell
git clone <repo> ; cd cctv-forall
.\scripts\setup_windows.ps1          # GPU dideteksi otomatis -> profil nvidia
# edit .env (ADMIN_PASSWORD, JWT_SECRET)
.\scripts\run_windows.ps1

curl http://localhost:8000/api/health
# "device": "cuda", "backend": "torch", "half": true,
# "cuda_devices": [{"name": "NVIDIA GeForce RTX 5070 Laptop GPU", ...}]
```

Setup script memeriksa GPU **sebelum** vendor CPU. Laptop NVIDIA hampir selalu
ber-CPU AMD atau Intel, dan memilih profil dari CPU di situ akan menyiapkan
plugin CPU OpenVINO lalu membiarkan kartu diskritnya menganggur.

⚠️ **Wheel torch harus cocok dengan kartunya.** RTX seri 50 adalah sm_120
(Blackwell) dan build cu128 ke bawah tidak punya kernel untuknya — terpasang
tanpa keluhan, lalu gagal saat kernel pertama diluncurkan. Profil nvidia memasang
cu130. Kalau `/api/health` melaporkan `"torch": "...+cpu"`, yang terpasang masih
wheel CPU:

```powershell
backend\.venv\Scripts\python -m pip install --force-reinstall `
  torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Aplikasi menolak start dengan `DEVICE=cuda` di atas torch CPU-only, dengan pesan
yang menyebutkan perintah di atas — bukan gagal jauh di dalam ultralytics.

### Hasil pengukuran (RTX 5070 Laptop 8 GB + Ryzen 9 8940HX, 1080p, imgsz=640, deteksi saja)

| Model | torch CPU | OpenVINO CPU INT8 | **CUDA fp16** |
| --- | --- | --- | --- |
| yolo11s | 19.9 fps | 43.9 fps | **84.1 fps** |
| yolo11m | 7.3 fps | — | **90.5 fps** |

Baseline lama di mesin ini adalah kolom OpenVINO INT8 (43.9 fps, yolo11s).
Jadi pindah ke GPU memberi **~2× throughput sekaligus model yang lebih besar**.

Perhatikan yolo11m tidak lebih lambat dari yolo11s: di imgsz 640 keduanya sudah
mentok di sisi CPU, bukan di forward pass. Dari ~11 ms per frame, hanya sekitar
4 ms yang benar-benar inferensi — sisanya letterbox, NMS dan overhead Python
ultralytics. Itulah alasan profil ini memakai `yolo11m`: akurasinya lebih baik
dan praktis gratis. Itu juga alasan TensorRT bukan prioritas (lihat bawah).

Ukur di mesin sendiri:

```powershell
backend\.venv\Scripts\python scripts\benchmark.py `
  --model data\weights\yolo11m.pt --imgsz 640
```

### Pemakaian VRAM (penting di kartu 8 GB)

Setiap source jalan di prosesnya sendiri dengan modelnya sendiri, jadi biayanya
per-stream. Terukur lewat `torch.cuda.mem_get_info()`:

| Tahap | VRAM terpakai |
| --- | --- |
| Konteks CUDA kosong (per proses) | 1166 MiB |
| + yolo11m fp16 | 1330 MiB |
| + EasyOCR + detektor plat | 1388 MiB |
| Saat memproses frame | ~1482 MiB |

**Yang mendominasi adalah konteks CUDA (~1.17 GB per proses), bukan modelnya
(~300 MB).** Itu konsekuensi arsitektur satu-proses-per-source. Praktisnya di
8 GB: 2 stream ≈ 3 GB (lapang, ini default `EXPECTED_STREAMS=2`), 4 stream
≈ 6 GB (muat, tapi sisakan ruang untuk desktop Windows). Di atas itu perlu
inference server bersama, bukan sekadar tuning.

`Detector.trim_memory()` dipanggil tiap 120 detik dan mengembalikan ~100 MiB
per worker.

### Apa saja yang dioptimalkan

- **Deteksi kendaraan di CUDA fp16** — `DEVICE=cuda` + `INFERENCE_HALF=true`.
  fp16 gratis di tensor core Blackwell dan memangkas jejak bobot/aktivasi.
- **`FRAME_STRIDE=1`** — profil CPU melewati frame karena inferensinya mahal.
  Di sini tidak, dan setiap frame diproses berarti ByteTrack punya dua kali
  lipat titik untuk diasosiasikan. Itu yang membuat hitungan garis lebih andal
  pada kendaraan cepat.
- **OCR plat di GPU** — `OCR_DEVICE=cuda`. EasyOCR itu model torch (CRAFT +
  CRNN), jadi ikut pindah tanpa perubahan kode.
- **Detektor plat khusus** — `PLATE_MODEL` diarahkan ke fine-tune YOLO11
  ([morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection),
  diunduh otomatis oleh setup script). Tanpa ini `alpr.py` memakai heuristik
  "ambil 45% bawah box kendaraan", yang merupakan sumber salah-baca terbesar.
  Di GPU biayanya nyaris nol.
- **Percobaan ANPR lebih longgar** — `ALPR_MAX_ATTEMPTS=24`,
  `ALPR_ATTEMPT_INTERVAL=2` (profil CPU: 12 dan 3). Lebih banyak percobaan per
  kendaraan berarti lebih besar peluang menangkap satu frame di mana platnya
  tegak lurus dan terbaca.
- **Budget thread naik jadi 4, affinity dimatikan** — jepitan 2-thread itu tepat
  untuk akselerator terintegrasi, di mana CPU juga yang menyuapi mereka. Kartu
  diskrit lain: core benar-benar bebas, tapi worker masih resize, letterbox,
  anotasi dan encode JPEG di sana. `WORKER_CPU_AFFINITY=off` karena pinning ada
  khusus untuk mengikat TBB milik OpenVINO — worker CUDA tidak punya pool itu,
  jadi mengurungnya ke irisan core hanya memperlambat.

### Kenapa `FFMPEG_HWACCEL` tetap `auto`, bukan `cuda`

Jangan set `cuda`. FFmpeg yang dibundel di dalam `opencv-python`
(`opencv_videoio_ffmpeg4110_64.dll`) **tidak punya decoder NVDEC** — scan DLL-nya
memberi 0 kecocokan untuk `h264_cuvid` dan `hevc_cuvid`, sementara `d3d11va`
muncul 15 kali. Selain itu `hwaccel` adalah opsi FFmpeg *CLI* yang tidak
dimengerti `avformat_open_input`, jadi jalur itu praktis no-op.

`auto` memilih D3D11VA, yang sudah memindahkan decoding ke media engine dan
membebaskan core CPU. NVDEC baru masuk akal kalau pipeline decode-nya diganti
(PyAV atau ffmpeg dengan nv-codec-headers), dan itu bukan tuning konfigurasi.

### Face recognition di GPU (ONNX Runtime)

`cv2.dnn` tidak punya backend CUDA di build OpenCV dari pip, jadi YuNet/SFace
dijalankan lewat **ONNX Runtime** — model yang sama persis dari
`data/weights/face/`, tanpa export ulang. Dipasang oleh setup script via
`backend/requirements-cuda.txt`.

Satu frame 1280x720 berisi 5 wajah, deteksi + embedding:

| Backend | Waktu |
| --- | --- |
| `cv2.dnn` (lama) | 86.9 ms |
| ONNX Runtime CPU | 61.8 ms |
| **ONNX Runtime CUDA** | **16.0 ms** |

Sebelum perubahan ini face recognition adalah **langkah termahal di seluruh
pipeline** — 86.9 ms, sekitar 8× biaya deteksi kendaraan (11 ms).

Yang membawa keuntungan terbesar di sini adalah **GPU**-nya: dari total 5.4×,
sekitar 3.9× datang dari pindah ke CUDA dan hanya 1.4× dari ganti runtime.
`FACE_BACKEND=auto` tetap memilih ONNX Runtime di mesin tanpa GPU (masih 1.4×
lebih cepat), dan otomatis mundur ke OpenCV kalau onnxruntime tidak terpasang.

Tidak perlu instalasi CUDA Toolkit terpisah: `onnxruntime-gpu` 1.29 me-link ke
CUDA 13 + cuDNN 9, yang persis sudah dibawa wheel torch `+cu130` di
`torch/lib`; `onnx_face.py` mengarahkan loader ke sana.

⚠️ **Kedua backend menghasilkan embedding yang BERBEDA.** Dengan blob input
yang byte-identik, cosine keduanya ~0.93 (bukan 1.0) — `cv2.dnn` menyimpang
dari semantik ONNX; ORT di CPU dan di CUDA justru sama persis. Pengenalan
membandingkan embedding tersimpan dengan embedding live, jadi **keduanya wajib
dari backend yang sama**. Kalau Anda sudah punya wajah terdaftar dan mengubah
`FACE_BACKEND`, **daftarkan ulang semua wajah**.

Decoding anchor YuNet ditulis ulang dengan tangan di `onnx_face.py`, dan itu
bagian paling berisiko dari perubahan ini — salah decode tidak error, hanya
mendeteksi hal yang salah. Karena itu ada validatornya:

```powershell
backend\.venv\Scripts\python scriptsalidate_onnx_face.py
```

Ia memberi tensor yang identik ke kedua implementasi dan gagal kalau kotak,
landmark atau skornya menyimpang lebih dari 0.5 px (hasil sekarang: 0.03 px).

### TensorRT (opsional, dan kemungkinan besar tidak perlu)

`scripts/export_tensorrt.py` ada kalau ingin memerasnya lebih jauh:

```powershell
backend\.venv\Scripts\python -m pip install -r backend\requirements-cuda.txt
backend\.venv\Scripts\python scripts\export_tensorrt.py --model data\weights\yolo11m.pt --imgsz 640
```

Tapi ukur dulu. Hanya ~4 ms dari ~11 ms per frame yang berupa inferensi, jadi
TensorRT cuma bisa memampatkan bagian itu. Engine-nya juga terikat pada GPU,
versi driver dan versi TensorRT — update driver rutin saja sudah memaksa build
ulang. Simpan `.pt`-nya sebagai salinan portabel.

---

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
5. Buka tab **Atur Zona Baca Plat** → **tarik sebuah kotak** pada bagian gambar
   tempat plat paling jelas terbaca (biasanya area terdekat dengan kamera, di
   sekitar garis hitung) → **Simpan Zona**. Worker restart dengan zona baru.
6. Kembali ke tab **Live** — hitungan per kelas/arah bertambah saat kendaraan
   melintasi garis. Plat nomor terbaca muncul di panel & halaman **Plat Nomor**;
   klik gambarnya untuk melihat frame penuh kendaraan tersebut.

### Kenapa perlu zona baca?

Tanpa zona, kendaraan mulai dicoba dibaca sejak muncul di ujung frame — paling
kecil, paling tidak terbaca — sehingga jatah percobaannya (`ALPR_MAX_ATTEMPTS`)
habis justru sebelum ia cukup dekat. Di mana plat terbaca adalah sifat
**kamera**, bukan sifat model: tergantung jarak, sudut dan lensa, dan hanya
orang yang melihat gambarnya yang tahu di mana itu. Zona adalah cara Anda
memberi tahu program.

Zona diperiksa terhadap **titik kontak roda** kendaraan (tengah-bawah box),
bukan tengah box — karena titik tengah melayang makin tinggi pada kendaraan
makin besar, sehingga bus bisa lolos dari lajur sebelah. Frame yang di luar zona
**tidak memakai jatah percobaan**, jadi seluruh jatah tersisa untuk tempat yang
memang bisa berhasil.

## Konfigurasi (env)

| Variabel | Default | Keterangan |
|---|---|---|
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | admin / admin | kredensial login |
| `JWT_SECRET` | change-me | rahasia token — wajib diganti |
| `DEVICE` | *(auto)* | `cpu` / `cuda` / `mps` / `intel:cpu` / `intel:gpu` |
| `YOLO_MODEL` | yolo11n.pt | ganti ke `yolo11s.pt` dll untuk akurasi lebih |
| `FRAME_STRIDE` | 2 | proses 1 dari N frame (hemat CPU) |
| `INFERENCE_IMGSZ` | 640 | resolusi inference |
| `INFERENCE_HALF` | false | inference fp16 (khusus GPU, ukur dulu) |
| `PROCESS_WIDTH` | 0 | perkecil frame ke lebar ini sebelum deteksi (0 = asli) |
| `COUNT_HYSTERESIS` | 0.02 | jarak bebas dari garis (rasio tinggi frame) sebelum sisi dianggap tercapai; 0 = nonaktif |
| `THREADS_PER_WORKER` | 0 | thread CPU per worker (0 = otomatis dari jumlah P-core) |
| `EXPECTED_STREAMS` | 4 | perkiraan jumlah source aktif, untuk pembagian thread |
| `WORKER_CPU_AFFINITY` | `auto` | kunci worker ke irisan core sendiri; pakai `off` di CUDA |
| `FFMPEG_HWACCEL` | `auto` | decoding di hardware; `off` untuk software (lihat *Troubleshooting* untuk kamera H.265) |
| `RTSP_TRANSPORT_TCP` | true | paksa RTSP lewat TCP |
| `FFMPEG_LOG_SUMMARY_SECONDS` | 30 | rangkum pesan decoder FFmpeg jadi satu baris tiap N detik (0 = tampilkan mentah) |
| `YOUTUBE_MAX_HEIGHT` | 720 | batas resolusi stream YouTube |
| `CAPTURE_BUFFER_SECONDS` | 2 | jitter buffer untuk HLS/YouTube (RAM: ±83 MB/detik pada 720p) |
| `MAX_STREAM_LATENCY_SECONDS` | 0.5 | batas ketertinggalan dari live sebelum reader lompat ke frame terbaru (0 = jangan lompat) |
| `MJPEG_FPS` | 15 | frame per detik yang dikirim ke browser |
| `JPEG_QUALITY` | 70 | kualitas JPEG stream |
| `OCR_DEVICE` | *(auto)* | perangkat EasyOCR; MPS sengaja tetap di CPU |
| `CONF_THRESHOLD` | 0.35 | ambang confidence deteksi |
| `ALPR_ENABLED` | true | aktif/nonaktif ANPR global |
| `PLATE_MODEL` | *(kosong)* | bobot detektor plat; kosong = heuristik ROI (jauh kurang akurat) |
| `PLATE_IMGSZ` | 320 | resolusi detektor plat (jalan di crop kendaraan, bukan frame penuh) |
| `PLATE_CONF_THRESHOLD` | 0.25 | ambang confidence detektor plat |
| `PLATE_MIN_CONFIDENCE` | 0.20 | ambang confidence OCR; di bawah ini plat tidak disimpan |
| `ALPR_MAX_ATTEMPTS` | 12 | berapa frame satu kendaraan dikejar sebelum menyerah |
| `ALPR_ATTEMPT_INTERVAL` | 3 | jalankan OCR tiap N percobaan (1 = tiap frame deteksi) |
| `ALPR_CLASSES` | `car,truck,bus` | kelas yang platnya dibaca; motor sengaja tidak termasuk |
| `ALPR_MIN_VEHICLE_WIDTH` | 160 | lebar minimum box kendaraan (piksel) sebelum plat dicoba dibaca |
| `ALPR_SAVE_FRAME` | true | simpan 1 frame penuh (kendaraan dikotaki) per plat terbaca |
| `FACE_ENABLED` | true | aktif/nonaktif face recognition global |
| `FACE_SIMILARITY_THRESHOLD` | 0.363 | ambang cosine SFace (lebih tinggi = lebih ketat) |
| `FACE_BACKEND` | `auto` | `onnx` (GPU, cepat) / `opencv` / `auto`; ganti = wajib daftar ulang wajah |
| `FACE_MAX_SIDE` | 1024 | frame diperkecil ke sisi terpanjang ini sebelum deteksi wajah |
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
| `masuk` rendah **dan `deteksi` jauh lebih rendah** | sumbernya yang lambat — kamera/jaringan/YouTube | bukan masalah server; cek jaringan atau ganti sumber |
| `masuk` ≈ `deteksi`, keduanya rendah, `dibuang` = 0 | **loopnya yang jadi rem, bukan sumbernya** — lihat catatan di bawah | naikkan `FRAME_STRIDE`, set `PROCESS_WIDTH`, atau pakai model lebih kecil |
| `masuk` normal, `deteksi` rendah | inference tidak terkejar | naikkan `FRAME_STRIDE`, set `PROCESS_WIDTH`, atau pakai model lebih kecil |
| `dibuang` tinggi pada sumber live | pemrosesan lebih lambat dari stream; frame lama dibuang agar tetap di live edge | naikkan `FRAME_STRIDE` atau pakai model lebih kecil |
| `tampil` jauh di bawah `masuk` | batas `MJPEG_FPS` | naikkan `MJPEG_FPS` (ideal ≥ fps kamera) |
| `tampil` normal tapi `deteksi` sangat rendah | preview tetap live, yang tersendat inference/OCR | lihat catatan ANPR — turunkan `ALPR_MAX_ATTEMPTS`, naikkan `ALPR_ATTEMPT_INTERVAL`, atau pindahkan OCR ke CPU |

Lewat API: `curl -s localhost:8000/sources -H "authorization: Bearer <token>"`.

Beberapa hal yang perlu diketahui:

- **Hitungan memakai titik kontak roda, bukan titik tengah box.** Titik tengah
  melayang makin tinggi pada kendaraan makin besar — bus dan motor di posisi
  jalan yang sama akan melintasi garis pada saat berbeda (terukur 14 frame
  bedanya). Titik tengah juga mewarisi separuh goyangan **tepi atas** box, dan
  tepi atas justru yang paling tidak stabil: melompat saat model
  memasukkan/mengeluarkan kabin, kontainer, atau spion. Roda tidak ke mana-mana.
- **`COUNT_HYSTERESIS` mencegah hitungan ganda.** Sebuah sisi baru dianggap
  tercapai setelah kendaraan bebas dari garis sejauh nilai ini; di dalam pita
  itu sisinya belum ditentukan sehingga goyangan tidak mendaftarkan apa pun.
  Jangan diisi besar: nilai ini juga jarak yang masih harus ditempuh kendaraan
  **setelah** melintas, jadi garis yang digambar sangat dekat tepi frame dengan
  pita lebar bisa melewatkan kendaraan yang keburu keluar layar.
- **`MJPEG_FPS` sebaiknya ≥ fps kamera.** Frame hanya datang pada irama kamera,
  jadi kamera 30 fps dengan `MJPEG_FPS=20` akan jatuh ke 15 fps. Isi 30 untuk
  video mulus, atau turunkan ke 10–15 untuk menghemat CPU saat banyak tile.
- **Kemulusan video tidak terikat kecepatan deteksi.** Preview punya thread
  sendiri yang membaca frame terbaru dari decoder dan menggambar ulang kotak
  terakhir, jadi `FRAME_STRIDE=2` tidak membuat video setengah lambat — dan
  deteksi yang tersendat (model berat, ANPR rakus) tidak lagi membekukan
  tampilan. Angka `deteksi` pada panel statistik yang menunjukkan hal itu,
  bukan gambarnya.
- **`masuk` bukan pengukur kamera pada sumber HLS/YouTube.** Decoder di jalur
  itu direm oleh buffer (backpressure), jadi kalau loopnya lambat, `masuk` ikut
  turun sampai sama dengan `deteksi` — seolah-olah sumbernya yang lambat.
  Tandanya: `masuk` ≈ `deteksi` **dan** `dibuang` = 0. Untuk memastikan, ukur
  decoding murni dengan `ALPR_ENABLED=false` dan `FRAME_STRIDE` besar; kalau
  `masuk` melonjak, batasnya ada di pemrosesan.
- **ANPR jalan di thread sendiri.** Membaca plat = satu pass detektor plat +
  satu pass OCR per kendaraan, dan simpang ramai punya beberapa kendaraan belum
  terbaca sekaligus. Kalau dijalankan di dalam loop, biaya itu menentukan
  kecepatan seluruh pipeline. Crop-nya sekarang diserahkan ke thread terpisah
  dan hasilnya menyusul sepersekian detik kemudian, jadi `ALPR_MAX_ATTEMPTS`
  yang besar tidak lagi menurunkan fps video.
- **Jatah percobaan hanya dipakai saat kendaraan cukup besar.** Sebuah kendaraan
  mulai dilacak sejak muncul di ujung frame — paling kecil, paling tidak
  terbaca. Tanpa `ALPR_MIN_VEHICLE_WIDTH`, jatahnya habis di sana dan kendaraan
  menyerah tepat sebelum cukup dekat untuk dibaca. Frame di bawah ambang itu
  **tidak memakai jatah**, jadi naikkan `ALPR_MAX_ATTEMPTS` bila perlu:
  percobaannya sekarang jatuh di frame yang memang bisa berhasil.
- **Yang disimpan adalah bacaan terbaik, bukan yang pertama.** Satu kendaraan
  dibaca beberapa kali sambil mendekat; bacaan dengan confidence lebih tinggi
  memperbarui baris yang sama (bukan menambah baris baru), dan crop lamanya
  dihapus. Tanpa ini, salah baca confidence 0.2 dari kejauhan terkunci permanen
  dan memblokir bacaan bagus yang datang dua detik kemudian di depan kamera.
- **YouTube live sering tersendat dari sananya.** Diukur pada stream CCTV live
  (hanya tersedia HLS): tanpa deteksi sama sekali, hanya decoding, tetap ada
  11–12 jeda >0.5 detik per 75 detik dengan jeda terpanjang **14 detik** — sama
  saja di 360p maupun 720p. Itu batas sumbernya, bukan Mac-nya. **RTSP langsung
  dari CCTV jauh lebih stabil** dan itu jalur yang dioptimalkan di sini.
- **`CAPTURE_BUFFER_SECONDS`** memperhalus HLS yang datang per segmen, tapi
  menyimpan frame mentah: ±83 MB per detik buffer pada 720p. Menaikkannya di
  atas ±4 detik jarang sepadan.
- **`MAX_STREAM_LATENCY_SECONDS`** adalah pasangannya. Jitter buffer melepas
  frame pada irama stream, dan itu hanya benar selama deteksi terkejar; kalau
  tidak, video live jadi *slow motion* yang makin lama makin tertinggal. Angka
  ini adalah batas ketertinggalan itu — lewat dari situ, frame yang menumpuk
  dibuang dan pemutaran lanjut dari yang terbaru. Biarkan kecil: mengejar
  ketertinggalan berarti melompati frame di antaranya, jadi angka besar bukan
  membuat mulus, hanya menukar beberapa lompatan kecil dengan satu lompatan
  besar. Isi 0 hanya untuk sumber rekaman, saat memproses semua frame lebih
  penting daripada tetap aktual.

### Log dibanjiri `Could not find ref with POC N` (kamera 4K H.265)

```
[hevc @ 0000023dae41de40] Could not find ref with POC 10
[hevc @ 0000023dae428d80] Could not find ref with POC 12
[hevc @ 0000023b3398e940] Could not find ref with POC 36
```

Artinya **ada frame yang hilang sebelum sampai ke decoder**: decoder diminta
menyusun sebuah frame dari frame referensi yang tidak pernah datang. Bedakan
dua penyebabnya, karena obatnya berlawanan:

- **Jaringan/kamera** — kabel, Wi-Fi, switch, atau RTSP lewat UDP. Lihat bagian
  di bawah.
- **Mesinnya yang tidak terkejar** — ini yang paling sering pada kamera 4K
  H.265. Selama worker sibuk, socket RTSP tidak sempat dikosongkan, buffer
  kirim di kamera penuh, dan **kamera membuang frame yang tidak bisa ia
  kirim**. Yang terlihat di log adalah akibatnya, bukan sebabnya.

Cara memastikan yang mana: **matikan source-nya, lalu decode saja stream yang
sama.** Kalau bersih, jaringannya tidak bersalah. Diukur pada RTX 5070 Laptop +
Ryzen 9 8940HX, Hikvision 3840×2160 H.265 25 fps, `PROCESS_WIDTH=1280`,
yolo11m + ANPR + face aktif:

| Kondisi | `masuk` (fps) | `deteksi` (fps) | Pesan decoder / menit |
|---|---|---|---|
| decoding saja, worker mati | 25.3 | — | **0** |
| worker penuh, main stream 3840×2160 | 20–23.5 | 15–17 | 13–35 |
| worker penuh, substream 1280×720 | **25.0** | **24.4** | **0** |

Urutan penanganannya:

1. **Pakai substream kamera untuk deteksi.** Pada Hikvision cukup ganti akhiran
   URL `/Streaming/Channels/101` (main) menjadi `/Streaming/Channels/102`
   (sub). Ini yang menghilangkan masalahnya sekaligus menaikkan `deteksi` dari
   ±16 ke ±24 fps. *Catatan ANPR:* crop plat diambil dari **frame penuh**, jadi
   4K memang membantu OCR plat jauh — kalau plat harus terbaca dari kejauhan,
   pilih opsi 2 dulu.
2. **Turunkan beban stream di setting kamera.** Frame rate 25 → 12–15 fps sudah
   lebih dari cukup untuk deteksi dan langsung memotong separuh biaya decoding,
   tanpa mengorbankan resolusi yang dibutuhkan ANPR. Menurunkan bitrate main
   stream juga membantu (biaya decoding H.265 mengikuti isi gambar — jam sibuk
   lebih mahal daripada jalan sepi, yang membuat gejalanya datang-pergi).
3. **`FRAME_STRIDE` dan `PROCESS_WIDTH` tidak menolong di sini.** Keduanya
   mengurangi biaya *inference*; setiap frame tetap harus di-decode utuh.
4. **Jangan berharap dari `FFMPEG_HWACCEL`.** Diukur di mesin yang sama saat
   CPU sedang sibuk, D3D11VA justru **lebih lambat** untuk 4K (±7,6 vs ±11,1
   fps): menyalin frame 4K dari GPU kembali ke RAM lebih mahal daripada
   decoding yang dihematnya.

Sejak versi ini gejalanya juga dilaporkan, bukan cuma dimuntahkan ke konsol:

- Status source di dashboard menampilkan peringatan (kuning, source tetap
  *running*): `receiving 20 of 25 fps at 3840x2160: frames are being dropped
  before they reach the decoder ...`
- Log tidak lagi banjir. Baris-baris FFMPEG dirangkum menjadi satu baris tiap
  `FFMPEG_LOG_SUMMARY_SECONDS` detik:
  `[ffmpeg] source 6: 41 decoder message(s) in 30s (last: Could not find ref with POC 44)`.
  Isi `FFMPEG_LOG_SUMMARY_SECONDS=0` bila ingin baris mentahnya kembali.

### Windows: `hardware accelerator failed to decode picture` (HEVC/H.265)

Log seperti ini muncul saat kamera H.265 di-decode oleh iGPU Intel:

```
[hevc @ ...] Could not find ref with POC 44
[hevc @ ...] Failed to execute: 0x80070057
[hevc @ ...] hardware accelerator failed to decode picture
```

Baris `Could not find ref with POC N` dan `Non-matching NAL types` berarti
**paket stream hilang** sebelum sampai ke decoder, jadi frame referensi yang
dibutuhkan tidak pernah datang. (Kalau yang muncul **hanya** baris itu, tanpa
`hardware accelerator failed`, kemungkinan besar bukan decoder-nya yang
menyerah melainkan mesinnya yang tidak terkejar — lihat bagian di atas.) Decoder software cuma menampilkan artefak dan
jalan terus; D3D11VA/DXVA langsung menyerah dengan `0x80070057` (E_INVALIDARG),
capture-nya mati, lalu reconnect di tengah GOP — dan siklusnya berulang.

Urutan penanganannya:

1. **Pastikan RTSP lewat TCP** — `RTSP_TRANSPORT_TCP=true` (default). Lewat UDP,
   satu paket hilang saja sudah cukup memicu pola di atas.
2. **Set kamera ke H.264, bukan H.265.** Ini perbaikan yang paling manjur:
   decoder H.264 di HD Graphics jauh lebih tahan terhadap paket rusak, dan
   substream 720p H.264 sudah lebih dari cukup untuk deteksi.
3. **Matikan hardware decoding:** `FFMPEG_HWACCEL=off`. Decoding pindah ke CPU
   (mahal di mesin kecil, tapi tahan terhadap frame rusak) dan iGPU-nya bebas
   penuh untuk inference OpenVINO.
4. Kalau jaringannya memang bermasalah, cek kabel/switch/Wi-Fi kamera dan
   turunkan bitrate substream.

Sejak versi ini, langkah 3 juga berjalan otomatis: bila hardware decoder gagal
mengirim frame dua kali berturut-turut, source pindah sendiri ke software
decoding sampai worker di-restart, dan log akan mencatat
`switching to software decoding`.

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
