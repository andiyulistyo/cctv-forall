# Arsitektur Detection Dashboard

Dokumen ini menjelaskan arsitektur lengkap sistem: proses, thread, aliran data,
pipeline deteksi, skema database, dan jalur inferensi per perangkat keras.

Semua diagram ditulis dalam **Mermaid**, jadi langsung ter-render di GitHub,
GitLab, dan preview Markdown VS Code. Untuk membukanya sebagai halaman HTML
biasa, lihat [Membuka sebagai HTML](#membuka-sebagai-html) di bagian akhir.

---

## 1. Arsitektur keseluruhan

Satu proses FastAPI melayani API + SPA, dan menyalakan **satu proses worker per
source**. Kedua sisi bertukar data lewat `multiprocessing.Manager`, bukan lewat
database — frame JPEG terlalu panas untuk lewat SQLite.

```mermaid
flowchart TB
    SRC["🎥 Sumber video<br/>YouTube · RTSP · RTMP · HLS · HTTP/MJPEG · File"]

    subgraph client["Browser"]
        SPA["React 18 SPA<br/>Vite · TypeScript · Tailwind 3<br/>React Router 6"]
    end

    subgraph apiproc["Proses FastAPI — uvicorn"]
        REST["REST API + JWT<br/>/api/auth · /api/sources · /api/counts<br/>/api/plates · /api/faces · /api/sightings"]
        MJPEG["MJPEG endpoint<br/>/api/streams/{source_id}"]
        MGR["DetectionManager<br/>supervisor proses"]
        SCHED["APScheduler<br/>purge retensi berkala"]
        STATIC["StaticFiles<br/>serve frontend/dist"]
    end

    subgraph shared["SharedState — multiprocessing.Manager"]
        SH["frames · frame_seq<br/>live_counts · status · stats"]
    end

    subgraph workers["Worker proses — 1 per source, start method spawn"]
        W1["worker-1"]
        W2["worker-2"]
        WN["worker-N"]
    end

    subgraph store["Penyimpanan lokal — folder ./data"]
        DB[("SQLite app.db<br/>mode WAL")]
        IMG["data/plates/*.jpg<br/>data/faces/*.jpg"]
        WGT["data/weights/<br/>yolo11m.pt · plate .pt<br/>yunet.onnx · sface.onnx"]
    end

    SPA -->|"REST + Bearer JWT"| REST
    SPA -->|"multipart/x-mixed-replace"| MJPEG
    SPA --> STATIC

    REST -->|"start / stop"| MGR
    MGR -->|"Process + stop Event + slot CPU"| W1
    MGR --> W2
    MGR --> WN

    SRC --> W1
    SRC --> W2
    SRC --> WN

    W1 -->|"JPEG beranotasi + counter + status"| SH
    W2 --> SH
    WN --> SH
    MJPEG -->|"poll frame_seq, tarik JPEG bila berubah"| SH
    REST -->|"status live"| SH

    WGT --> W1
    W1 -->|"CountEvent · PlateRead · FaceSighting"| DB
    W1 --> IMG
    REST <--> DB
    SCHED -->|"hapus baris + file kedaluwarsa"| DB
    SCHED --> IMG
```

**Kenapa proses terpisah, bukan thread?** GIL Python akan membuat 4–10 stream
saling menunggu di loop deteksi. `spawn` dipakai (bukan `fork`) karena wajib di
Windows dan paling aman saat torch/CUDA di-load di dalam worker.

---

## 2. Anatomi satu worker

Satu source = satu proses, dan di dalamnya ada **tiga thread pembantu** supaya
loop deteksi tidak pernah menunggu I/O, OCR, atau encoding JPEG.

```mermaid
flowchart TB
    NET["Stream jaringan / file"]

    subgraph proc["Proses worker — satu source"]
        direction TB

        subgraph tdec["Thread decoder"]
            SR["SourceReader<br/>cv2.VideoCapture · backend FFmpeg<br/>hwaccel · RTSP over TCP · auto-reconnect"]
            BUF["BufferedFrameReader<br/>jitter buffer 2 dtk untuk HLS/YouTube<br/>lompat ke frame terbaru bila tertinggal"]
        end

        subgraph tmain["Thread utama — loop deteksi"]
            STRIDE{"frame_no % frame_stride == 0?"}
            SKIP["Lewati inferensi<br/>frame tetap masuk preview"]
            SCALE["Downscale ke process_width<br/>simpan detect_scale"]
            DET["Detector.track()<br/>Ultralytics YOLO11 + ByteTrack"]
            LC["LineCounter<br/>titik kontak roda + pita histeresis"]
            FACE["FaceRecognizer<br/>YuNet deteksi → SFace embedding"]
        end

        subgraph talpr["Thread ALPR"]
            AQ["Antrian crop kendaraan<br/>dari frame resolusi penuh"]
            ALPR["Plate detector YOLO → EasyOCR<br/>normalisasi + regex + ambang confidence"]
        end

        subgraph tpub["Thread publisher"]
            DRAW["Gambar box, garis, label,<br/>counter, nama wajah, teks plat"]
            ENC["cv2.imencode('.jpg')"]
        end
    end

    DBW[("SQLite")]
    SHM["SharedState → endpoint MJPEG"]

    NET --> SR --> BUF --> STRIDE
    STRIDE -->|"tidak"| SKIP
    STRIDE -->|"ya"| SCALE --> DET
    DET --> LC
    DET --> FACE
    DET -->|"crop resolusi penuh"| AQ --> ALPR

    LC -->|"CountEvent"| DBW
    ALPR -->|"PlateRead + crop + frame bukti"| DBW
    FACE -->|"FaceSighting"| DBW

    SKIP --> DRAW
    LC --> DRAW
    FACE --> DRAW
    ALPR --> DRAW
    DRAW --> ENC --> SHM
```

Detail yang tercermin di kode:

- **Deteksi jalan tiap `frame_stride` frame**, tapi preview tetap menerima semua
  frame — video terlihat mulus walau inferensi lebih jarang.
- **ANPR meng-crop dari frame resolusi penuh**, bukan dari frame yang sudah
  di-downscale, supaya teks plat tetap terbaca. `detect_scale` yang mengembalikan
  koordinat box ke skala penuh.
- **Publisher adalah thread sendiri** karena anotasi + `imencode` cukup mahal dan
  tidak boleh menahan frame berikutnya.
- `frame_seq` (dict kecil) dipisah dari `frames` (puluhan KB) supaya klien MJPEG
  bisa cek "berubah tidak?" tanpa menarik JPEG lewat pickle round-trip.
- Line counting memakai **titik kontak roda**, bukan titik tengah box, plus pita
  histeresis — tanpa itu box yang bergoyang di dekat garis menghasilkan 8,5%
  hitungan berlebih untuk mobil dan 18,5% untuk truk pada feed uji.

---

## 3. Pipeline ANPR (plat nomor Indonesia)

Dua tahap: **lokalisasi** lalu **OCR**, dengan dua lapis validasi setelahnya.

```mermaid
flowchart TB
    IN["Box kendaraan hasil tracking<br/>class ∈ ALPR_CLASSES (car, truck, bus)"]
    W{"Lebar box cukup<br/>dan berada di dalam zona ALPR?"}
    ATT{"Percobaan track ini < ALPR_MAX_ATTEMPTS?<br/>tiap ALPR_ATTEMPT_INTERVAL frame"}
    CROP["Crop kendaraan dari frame penuh"]
    PM{"PLATE_MODEL diset?"}
    YOLO["YOLO plat khusus<br/>license-plate-finetune-v1s.pt<br/>imgsz 320 · conf 0.25<br/>ambil box confidence tertinggi"]
    HEUR["Heuristik ROI<br/>ambil 45% bagian bawah box"]
    UP{"Lebar ROI < 200 px?"}
    RESIZE["Upscale INTER_CUBIC ke 200 px"]
    OCR["EasyOCR readtext()<br/>latin · GPU bila tersedia"]
    NORM["normalize_plate()<br/>uppercase, buang non-A-Z0-9"]
    RE{"Cocok pola plat Indonesia?<br/>1-2 huruf, 1-4 digit, 1-3 huruf"}
    CONF{"confidence ≥ PLATE_MIN_CONFIDENCE?"}
    SAVE["Simpan PlateRead<br/>+ crop plat + frame bukti berkotak"]
    RETRY["Buang — coba lagi di frame berikutnya"]
    DROP["Berhenti untuk track ini"]

    IN --> W
    W -->|"tidak"| RETRY
    W -->|"ya"| ATT
    ATT -->|"habis"| DROP
    ATT -->|"masih ada"| CROP --> PM
    PM -->|"ya"| YOLO --> UP
    PM -->|"tidak"| HEUR --> UP
    UP -->|"ya"| RESIZE --> OCR
    UP -->|"tidak"| OCR
    OCR --> NORM --> RE
    RE -->|"tidak"| RETRY
    RE -->|"ya"| CONF
    CONF -->|"tidak"| RETRY
    CONF -->|"ya"| SAVE
```

**Kenapa dua lapis validasi?** Regex saja lemah — plat buram di kejauhan tetap
menghasilkan string yang *cocok polanya* tapi salah isinya. Karena plat salah
lebih buruk daripada tidak ada plat, hasil di bawah ambang confidence dibuang dan
kendaraan dicoba lagi pada frame yang lebih dekat.

**Kenapa motor dikecualikan secara default?** Plat motor Indonesia kira-kira
setengah lebar plat mobil, posisinya rendah dan sering menyudut. Pada feed
persimpangan contoh, motor adalah 77% kendaraan yang melintas dan menghasilkan
**nol** pembacaan — sambil menghabiskan 77% anggaran OCR milik mobil dan truk
yang sebenarnya terbaca. Tambahkan kembali bila kamera cukup dekat.

**Kenapa detector plat jauh lebih berharga daripada tuning OCR?** Heuristik "45%
bagian bawah" adalah sumber misread terbesar di pipeline ini; OCR yang bagus pun
tidak bisa menyelamatkan ROI yang salah.

---

## 4. Pipeline face recognition

```mermaid
flowchart LR
    F["Frame"] --> Y["YuNet — deteksi wajah<br/>baris 15 kolom:<br/>x, y, w, h, 10 landmark, score"]
    Y --> A["alignCrop 112×112<br/>berdasarkan 5 landmark"]
    A --> S["SFace — embedding 128-D"]
    S --> C["Cosine similarity<br/>vs galeri EnrolledFace"]
    C --> T{"≥ FACE_SIMILARITY_THRESHOLD<br/>default 0.363?"}
    T -->|"ya"| K["Label nama<br/>catat FaceSighting"]
    T -->|"tidak"| U["Label 'unknown'<br/>dicatat bila LOG_UNKNOWN"]
```

Dua backend menjalankan **model yang sama** dan bisa ditukar lewat `FACE_BACKEND`:

| Backend | Runtime | 1 frame 720p, 5 wajah (RTX 5070 Laptop) |
|---|---|---|
| `onnx` (disarankan) | ONNX Runtime, CUDA | **16.0 ms** |
| `onnx` tanpa GPU | ONNX Runtime, CPU | 61.8 ms |
| `opencv` (fallback) | `cv2.dnn`, CPU saja | 86.9 ms |

Sekitar 3,9× dari total 5,4× percepatan datang dari pindah ke GPU, sisanya dari
runtime-nya sendiri. Build pip OpenCV tidak punya backend CUDA sama sekali, jadi
`cv2.FaceDetectorYN` selamanya di CPU — itulah alasan `onnx_face.py` ada.

> ⚠️ **Embedding kedua backend tidak kompatibel.** Pada input byte-identik,
> cosine similarity keduanya ~0,93, bukan 1,0 — evaluasi SFace di `cv2.dnn`
> menyimpang dari ONNX Runtime. Mengganti backend berarti **semua wajah harus
> di-enroll ulang**.

---

## 5. Skema database

SQLite mode WAL. Retensi otomatis menghapus baris **dan** file gambarnya.

```mermaid
erDiagram
    USERS {
        int id PK
        string username UK
        string password_hash
    }
    SOURCES {
        int id PK
        string name
        string type "youtube rtsp rtmp hls http file"
        text url
        json enabled_classes
        json line "dua titik ternormalisasi"
        json direction_labels
        json alpr_zone
        bool alpr_enabled
        bool face_enabled
        string status
        text status_message
        datetime created_at
    }
    COUNT_EVENTS {
        int id PK
        int source_id FK
        string class_name
        string direction "in atau out"
        int track_id
        datetime timestamp
    }
    PLATE_READS {
        int id PK
        int source_id FK
        int track_id
        string vehicle_class
        string plate_text
        float confidence
        text image_path "crop plat"
        text frame_path "frame bukti"
        datetime timestamp
    }
    ENROLLED_FACES {
        int id PK
        string name
        json embedding "128 float"
        text image_path
        datetime created_at
    }
    FACE_SIGHTINGS {
        int id PK
        int source_id FK
        string name "null berarti unknown"
        float similarity
        text image_path
        datetime timestamp
    }

    SOURCES ||--o{ COUNT_EVENTS : menghasilkan
    SOURCES ||--o{ PLATE_READS : menghasilkan
    SOURCES ||--o{ FACE_SIGHTINGS : menghasilkan
    ENROLLED_FACES ||..o{ FACE_SIGHTINGS : "dicocokkan via embedding"
```

Setiap `PlateRead` menyimpan **dua gambar**: crop platnya dan satu frame penuh
dengan kendaraannya dikotaki — supaya bisa dipastikan plat itu menempel pada
kendaraan yang mana.

---

## 6. Alur permintaan: dari tambah source sampai stream live

```mermaid
sequenceDiagram
    autonumber
    actor U as Pengguna
    participant SPA as React SPA
    participant API as FastAPI
    participant DB as SQLite
    participant MGR as DetectionManager
    participant W as Worker proses
    participant SH as SharedState

    U->>SPA: Login (admin)
    SPA->>API: POST /api/auth/login
    API->>DB: cek user, verifikasi bcrypt
    API-->>SPA: JWT (HS256)

    U->>SPA: Tambah source (URL + kelas aktif)
    SPA->>API: POST /api/sources
    API->>DB: INSERT Source
    API-->>SPA: SourceOut

    U->>SPA: Gambar garis hitung & zona ALPR
    SPA->>API: GET /api/sources/{id}/snapshot
    API-->>SPA: JPEG satu frame
    SPA->>API: PUT /api/sources/{id}/line dan /alpr-zone
    API->>DB: simpan koordinat ternormalisasi

    U->>SPA: Start
    SPA->>API: POST /api/sources/{id}/start
    API->>MGR: start(source_cfg)
    MGR->>W: spawn Process(run_worker, shared, stop_event, slot)
    W->>W: pin CPU affinity, load YOLO / EasyOCR / YuNet
    W->>SH: set_status("running")
    API-->>SPA: status running

    loop Tiap frame
        W->>W: decode → deteksi → hitung → ANPR → wajah
        W->>DB: CountEvent / PlateRead / FaceSighting
        W->>SH: set_frame(JPEG, seq), set_counts, set_stats
    end

    SPA->>API: GET /api/streams/{id}?token=...
    activate API
    loop MJPEG
        API->>SH: get_frame_seq(id)
        alt seq berubah
            API->>SH: get_frame(id)
            API-->>SPA: boundary + JPEG
        end
    end
    deactivate API

    U->>SPA: Stop
    SPA->>API: POST /api/sources/{id}/stop
    API->>MGR: stop(id)
    MGR->>W: set stop_event, join, reap
```

Endpoint MJPEG adalah generator tanpa akhir, jadi `main.py` memasang signal
handler sendiri: `SHUTTING_DOWN` di-set tepat saat Ctrl-C tiba, karena uvicorn
menjalankan lifespan shutdown **setelah** semua respons in-flight selesai — dan
tanpa itu kedua sisi saling menunggu.

---

## 7. Pemilihan device inferensi

`plan_inference()` di `detector.py` adalah satu-satunya tempat keputusan ini
dibuat — worker memakainya untuk anggaran thread CPU, dan `/api/health`
melaporkannya.

```mermaid
flowchart TB
    S["Bobot model dari YOLO_MODEL"] --> D{"DEVICE diset eksplisit?"}
    D -->|"ya"| USE["Pakai apa adanya<br/>require_device() gagal cepat<br/>bila tidak tersedia"]
    D -->|"tidak"| F{"Format bobot?"}

    F -->|"*_openvino_model/ atau .xml"| OV{"Intel iGPU ada?"}
    OV -->|"ya"| OVG["intel:gpu"]
    OV -->|"tidak"| OVC["intel:cpu<br/>juga jalur cepat di AMD Zen 4<br/>AVX-512 / VNNI"]

    F -->|".engine"| TRT["TensorRT<br/>presisi terkunci saat build<br/>terikat GPU + driver ini"]
    F -->|".mlpackage / .mlmodel"| ANE["CoreML<br/>Apple Neural Engine"]
    F -->|".onnx"| ORT["ONNX Runtime<br/>provider dipilih sendiri"]

    F -->|".pt (torch)"| T{"torch.cuda tersedia?"}
    T -->|"ya"| CUDA["cuda<br/>+ INFERENCE_HALF untuk fp16"]
    T -->|"tidak"| M{"torch.backends.mps tersedia?"}
    M -->|"ya"| MPS["mps — Metal, Apple Silicon"]
    M -->|"tidak"| CPU["cpu"]
```

Anggaran CPU per worker (`runtime.py`): jumlah performance-core dibagi
`EXPECTED_STREAMS`, lalu tiap worker **di-pin ke irisan core-nya sendiri**.
Ini perlu karena env var jumlah thread tidak sampai ke semua backend — plugin
CPU OpenVINO menjadwal lewat TBB dan hanya menghormati affinity mask proses.

---

## 8. Peta modul

```mermaid
flowchart LR
    subgraph app["backend/app"]
        MAIN["main.py<br/>lifespan · CORS · signal handler"]
        CFG["config.py<br/>Settings (pydantic-settings)"]
        RT["runtime.py<br/>deteksi hardware · thread · affinity"]
        MOD["models.py — SQLAlchemy"]
        DBM["database.py — engine + WAL pragma"]
        RET["retention.py — purge APScheduler"]

        subgraph apis["api/"]
            A1["auth_routes · sources · counts"]
            A2["plates · faces · streams"]
        end

        subgraph det["detection/"]
            MG["manager.py — supervisor proses"]
            WK["worker.py — loop per source"]
            DT["detector.py — YOLO + ByteTrack"]
            AL["alpr.py — plat + EasyOCR"]
            FC["face.py — YuNet + SFace"]
            OF["onnx_face.py — backend ONNX Runtime"]
            LN["line_counter.py — crossing + histeresis"]
            SRD["source_reader.py — capture + yt-dlp"]
            FS["frame_store.py — SharedState"]
        end
    end

    MAIN --> apis
    MAIN --> RET
    apis --> MG --> WK
    WK --> DT
    WK --> AL
    WK --> FC
    WK --> LN
    WK --> SRD
    WK --> FS
    FC --> OF
    DT --> RT
    AL --> RT
    FC --> RT
    apis --> MOD --> DBM
    CFG --> MAIN
    CFG --> WK
```

---

## 9. Ringkasan tech stack

| Lapisan | Teknologi |
|---|---|
| Frontend | React 18, React Router 6, TypeScript 5, Vite 6, Tailwind 3 |
| API | FastAPI 0.115, Uvicorn 0.34, Pydantic 2, python-multipart |
| Auth | PyJWT (HS256), bcrypt — satu admin |
| Database | SQLAlchemy 2.0 + SQLite (WAL) |
| Penjadwalan | APScheduler 3.11 (retensi otomatis) |
| **Deteksi objek** | **Ultralytics 8.3.200 (YOLO11)** + **ByteTrack** (solver `lap` 0.5.12) |
| **Baca plat** | **YOLO plat khusus** untuk lokalisasi + **EasyOCR 1.7.2** untuk OCR |
| Wajah | **YuNet** (deteksi) + **SFace** (embedding 128-D), via OpenCV DNN atau ONNX Runtime |
| Computer vision | OpenCV 4.11 (headless + reguler), NumPy 2.1, Pillow 11 |
| Input video | OpenCV/FFmpeg; **yt-dlp** untuk YouTube |
| Akselerator | torch CUDA (cu130), Apple MPS + CoreML/ANE, OpenVINO (Intel iGPU / CPU), TensorRT, ONNX Runtime |
| Distribusi | Docker + Compose (varian CPU & GPU), script setup native Windows/macOS |

Kelas COCO yang dipakai: `person` (0), `car` (2), `motorcycle` (3), `bus` (5),
`truck` (7) — dipilih per source lewat `enabled_classes`.

---

## Membuka sebagai HTML

Tiga cara, dari yang paling cepat:

1. **VS Code** — buka file ini lalu `Ctrl+Shift+V` (Markdown Preview). Mermaid
   ter-render bila ekstensi *Markdown Preview Mermaid Support* terpasang.
2. **GitHub / GitLab** — push, lalu buka file-nya. Mermaid ter-render otomatis.
3. **HTML mandiri** — konversi sekali dengan pandoc:

   ```bash
   pandoc docs/ARCHITECTURE.md -s -o docs/architecture.html \
     --metadata title="Arsitektur Detection Dashboard"
   ```

   Karena pandoc tidak me-render Mermaid sendiri, tambahkan skripnya di HTML
   hasil konversi (satu tag `<script type="module">` yang mengimpor
   `mermaid.esm.min.mjs` lalu memanggil `mermaid.initialize({startOnLoad:true})`),
   atau pakai `mermaid-cli` untuk mengubah tiap blok jadi SVG lebih dulu.
