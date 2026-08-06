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
| `CONF_THRESHOLD` | 0.35 | ambang confidence deteksi |
| `ALPR_ENABLED` | true | aktif/nonaktif ANPR global |
| `RETENTION_DAYS` | 7 | umur data sebelum dihapus |

## Testing logika line counting

```bash
cd backend
PYTHONPATH=. python tests/test_line_counter.py
```

## Catatan performa & akurasi

- **CPU**: gunakan `yolo11n` + `FRAME_STRIDE` lebih besar untuk 4–10 stream.
- **GPU**: bisa model lebih besar & lebih banyak stream.
- **Akurasi ANPR** bergantung resolusi/sudut/pencahayaan CCTV; hasil disimpan
  beserta nilai confidence. ANPR bisa dimatikan per source untuk menghemat CPU.

## Troubleshooting

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
