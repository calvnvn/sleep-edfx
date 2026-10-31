# Automated Sleep Stage Classification using Multimodal Hybrid Deep Learning

Proyek akhir *Deep Learning* (Informatika, Universitas Multimedia Nusantara) untuk mengklasifikasikan lima tahap tidur AASM (Wake, N1, N2, N3, REM) dari sinyal polisomnografi pada dataset PhysioNet Sleep-EDF Expanded. Model yang diusulkan adalah Multi-Branch 1D-CNN (cabang EEG, EOG, EMG) yang digabung dengan 2-layer Bi-LSTM dan dilatih dengan Focal Loss. Evaluasi memakai *subject-wise* GroupKFold dengan metrik Macro-F1 dan Cohen's Kappa.

## Status saat ini

| Tahap | Status | Keterangan |
|---|---|---|
| Fase 0. Lingkungan dan struktur proyek | Selesai | Struktur folder, dependensi, panduan setup |
| EDA (`notebooks/01_eda.ipynb`) | Selesai | Profil dataset, rata-rata waktu per tahap, visualisasi kondisi data |
| **Fase 1. Pre-processing** (`notebooks/02_preprocessing.ipynb`) | **Sedang dikerjakan** | Bagian A sampai D selesai (inventaris, anatomi rekaman, label per epoch). Bagian E sampai J berikutnya |
| Fase 2. Baseline klasik | Belum | Hasil awal sudah ada di `results/ablation/` |
| Fase 3. Model deep learning pada SC-20 | Belum | |
| Fase 4. Benchmark SC-78 (*subject-wise* cross-validation) | Belum | |
| Fase 5. Ablasi | Belum | |
| Fase 6. Uji eksternal ST dan laporan | Belum | |

Rincian per fase ada di bagian [Progress dan roadmap](#7-progress-dan-roadmap).

## Daftar isi

1. [Gambaran proyek](#1-gambaran-proyek)
2. [Cepat mulai](#2-cepat-mulai)
3. [Persyaratan](#3-persyaratan)
4. [Setup langkah demi langkah](#4-setup-langkah-demi-langkah)
5. [Menjalankan proyek](#5-menjalankan-proyek)
6. [Struktur repositori](#6-struktur-repositori)
7. [Progress dan roadmap](#7-progress-dan-roadmap)
8. [Temuan sejauh ini](#8-temuan-sejauh-ini)
9. [Keputusan desain](#9-keputusan-desain)
10. [Dokumentasi](#10-dokumentasi)
11. [Panduan kolaborasi](#11-panduan-kolaborasi)
12. [Pemecahan masalah](#12-pemecahan-masalah)
13. [Dataset, lisensi, dan sitasi](#13-dataset-lisensi-dan-sitasi)
14. [Penulis](#14-penulis)

---

## 1. Gambaran proyek

### Alur pipeline

```
Berkas EDF mentah ──► EDA ──► Pre-processing ──► Cache .npz ──► Model ──► Evaluasi
(data/raw, 8,1 GB)   (01)    (02, Fase 1)       (data/processed) CNN+Bi-LSTM  MF1, κ
                                                                             subject-wise
```

| Komponen | Pilihan |
|---|---|
| Dataset | PhysioNet Sleep-EDF Expanded v1.0.0 |
| Kelas | Wake, N1, N2, N3 (gabungan Stage 3 dan 4), REM (standar AASM) |
| Satuan data | Epoch 30 detik. EEG dan EOG 100 Hz (3.000 sampel), EMG envelope 1 Hz (30 sampel) |
| Arsitektur | Multi-Branch 1D-CNN (EEG, EOG, EMG) + 2-layer Bi-LSTM |
| Loss | Focal Loss |
| Validasi | *Subject-wise* GroupKFold (pengelompokan berdasarkan ID subjek, bukan nama berkas) |
| Metrik | Macro-F1 (utama), Cohen's Kappa, F1 per kelas |

### Pembagian penggunaan data

| Kelompok | Rekaman | Subjek | Peran |
|---|---|---|---|
| **SC** (Sleep-Cassette) | 153 | 78 | Data utama pelatihan dan evaluasi. Subjek sehat, direkam di rumah (ambulatori) |
| SC-20 | 39 | 20 (subjek 00 sampai 19) | Subset untuk iterasi cepat dan prototipe |
| **ST** (Sleep-Telemetry) | 44 | 22 | Hanya uji generalisasi *out-of-distribution* (ada pengaruh Temazepam dan perangkat EMG berbeda) |

---

## 2. Cepat mulai

Untuk yang sudah terbiasa dengan Python, ringkasannya begini. Setiap langkah dijelaskan lengkap di [bagian 4](#4-setup-langkah-demi-langkah).

```bash
git clone <URL-REPOSITORI>
cd <nama-folder-repositori>
python -m venv .venv                       # Python 3.11
# aktifkan venv (Windows PowerShell: .\.venv\Scripts\Activate.ps1 | macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # atau cu126 untuk GPU NVIDIA
# unduh dataset (8,1 GB) ke data/raw/sleep-edfx/  -> lihat bagian 4.7
python scripts/edfx_eda.py                 # uji akhir: hasilnya harus identik dengan yang ada di repositori
```

Lalu buka `notebooks/01_eda.ipynb` dan `notebooks/02_preprocessing.ipynb` dengan kernel `.venv`.

---

## 3. Persyaratan

| Kebutuhan | Minimum | Catatan |
|---|---|---|
| Sistem operasi | Windows 10/11, macOS, atau Linux | Dikembangkan dan diuji di Windows 11 |
| Python | **3.11** | Diuji pada 3.11.9. Versi lain belum diuji |
| Git | Versi terbaru | |
| Ruang disk | ±10 GB bebas sekarang, ±20 GB bila seluruh cache Fase 1 disimpan | Data mentah 8,1 GB. Cache Fase 1 diperkirakan 3,5 sampai 7 GB (float16 atau float32) |
| RAM | 16 GB disarankan | Fase 1 memproses satu rekaman sekali waktu. Pelatihan nanti memuat cache ke memori |
| GPU | **Tidak wajib** sampai Fase 3 | EDA dan Fase 1 berjalan di CPU. Pelatihan dikembangkan pada NVIDIA RTX 3050 6 GB |
| Internet | Untuk mengunduh paket Python dan dataset | |

Kebutuhan per tahap:

| Tahap | Data mentah | Internet | GPU |
|---|---|---|---|
| Membaca hasil EDA (`01_eda.ipynb`) | Tidak (hasil sudah di repositori) | Tidak | Tidak |
| Menjalankan ulang EDA dan Fase 1 | Ya | Tidak | Tidak |
| Pelatihan model (Fase 3 ke atas) | Ya (atau cache Fase 1) | Tidak | Disarankan |

---

## 4. Setup langkah demi langkah

Seluruh perintah dijalankan dari **root repositori** (folder yang berisi `README.md`).

### 4.1 Mengambil repositori

```bash
git clone <URL-REPOSITORI>
cd <nama-folder-repositori>
```

Ganti `<URL-REPOSITORI>` dengan alamat repositori GitHub yang dibagikan kepada Anda.

### 4.2 Memeriksa Python

| Sistem | Perintah | Hasil yang diharapkan |
|---|---|---|
| Windows | `py -3.11 --version` | `Python 3.11.x` |
| macOS / Linux | `python3.11 --version` | `Python 3.11.x` |

Jika Python 3.11 belum ada, unduh dari [python.org](https://www.python.org/downloads/). Pada Windows, centang **Add python.exe to PATH** saat memasang.

### 4.3 Membuat lingkungan virtual

Lingkungan virtual menjaga paket proyek ini terpisah dari proyek lain di komputer Anda.

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Tanda lingkungan aktif: awalan `(.venv)` muncul di terminal. Seluruh perintah `python` dan `pip` berikutnya memakai lingkungan ini. Setiap membuka terminal baru, aktifkan ulang dengan perintah `activate` di atas.

> Jika PowerShell menolak `Activate.ps1` (pesan *running scripts is disabled*), ada dua jalan. Pertama, izinkan skrip untuk akun Anda sendiri dengan `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. Kedua, lewati aktivasi dan panggil Python langsung, misalnya `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

### 4.4 Memasang dependensi

```bash
python -m pip install -r requirements.txt
```

Berkas `requirements.txt` hanya memuat dependensi langsung (NumPy, SciPy, pandas, Matplotlib, scikit-learn, MNE, tqdm, ipykernel, JupyterLab) dan berlaku di semua sistem operasi. PyTorch **tidak** ada di dalamnya karena versinya bergantung pada perangkat Anda (langkah berikutnya).

### 4.5 Memasang PyTorch

Pilih **satu** jalur sesuai perangkat Anda.

| Perangkat | Perintah |
|---|---|
| GPU NVIDIA (Windows atau Linux), CUDA 12.6 | `python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126` |
| Tanpa GPU (CPU saja) | `python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu` |
| macOS | `python -m pip install torch==2.14.1` |

Catatan:

- Untuk EDA dan Fase 1, versi CPU sudah cukup. Anda dapat menukar ke versi GPU nanti dengan `python -m pip uninstall torch` lalu memasang ulang.
- Jalur GPU memerlukan driver NVIDIA yang cukup baru (disarankan seri 560 atau lebih baru). Periksa dengan `nvidia-smi`.
- Jalur CPU dan jalur GPU Windows sudah diuji. Jalur macOS dan Linux belum diuji pada perangkat sungguhan.

### 4.6 Memverifikasi lingkungan

```bash
python -c "import torch, numpy, scipy, pandas, matplotlib, sklearn, mne; print('torch', torch.__version__, '| GPU tersedia:', torch.cuda.is_available()); print('numpy', numpy.__version__, '| pandas', pandas.__version__)"
```

Keluaran yang diharapkan: `torch 2.14.1...`, lalu `numpy 2.4.6 | pandas 3.1.0`. Nilai `GPU tersedia` adalah `True` hanya bila Anda memakai jalur GPU. `False` pada jalur CPU adalah normal.

### 4.7 Mengunduh dataset

Dataset [Sleep-EDF Database Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) berukuran **8,1 GB**, terbuka untuk umum, dan **tidak disertakan di repositori ini**. Unduh **seluruhnya** (termasuk Sleep-Telemetry), karena skrip `scripts/edfx_eda.py` membaca keduanya.

Pilih salah satu cara:

**A. AWS CLI (semua sistem operasi, mendukung unduhan lanjutan).** Pasang [AWS CLI](https://aws.amazon.com/cli/) (gratis), lalu:

```bash
aws s3 sync --no-sign-request s3://physionet-open/sleep-edfx/1.0.0/ data/raw/sleep-edfx
```

Tidak memerlukan akun AWS. Perintah ini membuat folder tujuan sendiri dan melanjutkan bila terputus. Pada pengujian, perintah ini mendaftar 398 berkas, sama dengan jumlah baris `SHA256SUMS.txt`.

**B. `wget` (Linux, macOS, atau WSL; belum diuji di mesin pengembang karena `wget` tidak terpasang).**

```bash
wget -r -N -c -np -nH --cut-dirs=3 -R "index.html*" -P data/raw/sleep-edfx https://physionet.org/files/sleep-edfx/1.0.0/
```

**C. Peramban (tanpa alat tambahan).** Unduh berkas ZIP (8,1 GB) dari [halaman dataset](https://physionet.org/content/sleep-edfx/1.0.0/), lalu ekstrak isinya ke `data/raw/sleep-edfx/`.

**Tata letak yang benar** setelah unduhan selesai (isi folder harus langsung berada di bawah `data/raw/sleep-edfx/`, tanpa folder perantara seperti `1.0.0/`):

```
data/raw/sleep-edfx/
├── RECORDS
├── RECORDS-v1
├── SC-subjects.xls
├── ST-subjects.xls
├── SHA256SUMS.txt
├── sleep-cassette/       # 153 berkas *-PSG.edf + 153 berkas *-Hypnogram.edf
└── sleep-telemetry/      # 44 berkas *-PSG.edf + 44 berkas *-Hypnogram.edf
```

**Verifikasi keutuhan data** (disarankan, ±35 detik). Semua berkas harus lolos tanpa baris `FAILED`:

```bash
cd data/raw/sleep-edfx
sha256sum -c --quiet SHA256SUMS.txt      # Linux, atau Git Bash di Windows
# macOS: shasum -a 256 -c SHA256SUMS.txt
cd ../../..
```

Jika ada berkas yang `FAILED`, unduh ulang berkas itu saja.

> Folder `data/raw/` diabaikan git dan **tidak boleh diubah** oleh kode apa pun. Seluruh hasil olahan disimpan di `data/processed/`.

### 4.8 Menyiapkan editor

**VS Code (disarankan).** Pasang ekstensi **Python** (`ms-python.python`) dan **Jupyter** (`ms-toolsai.jupyter`), lalu buka folder root repositori. Saat membuka notebook, klik **Select Kernel** di kanan atas dan pilih interpreter `.venv` (`Python 3.11.x ('.venv')`). Jika tidak muncul: `Ctrl+Shift+P`, ketik `Python: Select Interpreter`, pilih `.venv`, lalu muat ulang jendela.

**JupyterLab (alternatif tanpa VS Code).** Dengan lingkungan aktif:

```bash
jupyter lab
```

### 4.9 Uji akhir: apakah semuanya bekerja?

Jalankan skrip EDA, yang membaca header dan hypnogram ke-197 rekaman:

```bash
python scripts/edfx_eda.py
```

Berkas hasilnya (`results/eda/eda_summary.txt` dan `eda_records.json`) sudah ada di repositori. Jika data Anda utuh, hasil baru **identik** dengan yang tersimpan, sehingga perintah berikut tidak menampilkan apa pun:

```bash
git status --short results/eda
```

Jika berkas itu muncul sebagai `M` (berubah), data Anda berbeda dari yang digunakan di proyek ini; periksa kembali langkah verifikasi 4.7.

---

## 5. Menjalankan proyek

Jalankan notebook **berurutan**, dengan kernel `.venv`. Notebook mencari root repositori sendiri, sehingga dapat dijalankan dari folder mana pun di dalam repositori.

| Berkas | Prasyarat | Keluaran | Keterangan |
|---|---|---|---|
| `scripts/edfx_eda.py` | Dataset lengkap | `results/eda/eda_summary.txt`, `eda_records.json` | Ringkasan per rekaman. Hanya membaca header dan hypnogram, tidak membaca sinyal |
| `notebooks/01_eda.ipynb` | `results/eda/eda_records.json` (sudah di repositori) | 4 grafik di `results/figures/`, 1 tabel di `results/tables/` | Pada akhir sel 3 tercetak `All checks passed` |
| `notebooks/02_preprocessing.ipynb` | Dataset lengkap | Grafik `02_*` di `results/figures/` | Sel D4 mencetak `153 recordings, 416,424 epochs; all counts match the EDA` |

Estimasi waktu: TBD_WAKTU

Panduan membaca tiap sel (analogi, cara membaca keluaran, pertanyaan latihan) ada di `docs/belajar/`.

---

## 6. Struktur repositori

```
.
├── configs/                  # konfigurasi percobaan (YAML), satu berkas per eksperimen (belum dipakai)
├── data/
│   ├── raw/sleep-edfx/       # dataset PhysioNet, hanya dibaca (tidak di-commit)
│   └── processed/            # cache hasil pre-processing (tidak di-commit)
├── docs/
│   ├── analisis/             # analisis proyek secara menyeluruh
│   ├── belajar/              # bahan belajar per notebook (01 dan 02)
│   └── catatan/              # catatan perubahan Ref-1 dan keputusan desain
├── notebooks/
│   ├── 01_eda.ipynb          # EDA
│   └── 02_preprocessing.ipynb  # Fase 1 (sedang dikerjakan)
├── results/
│   ├── eda/                  # ringkasan EDA (eda_records.json, eda_summary.txt)
│   ├── ablation/             # hasil baseline klasik awal
│   ├── figures/              # grafik (awalan 01_ dan 02_ mengikuti notebook)
│   ├── tables/               # tabel hasil
│   ├── logs/                 # log pelatihan (belum dipakai)
│   └── checkpoints/          # bobot model (tidak di-commit)
├── scripts/
│   └── edfx_eda.py           # pembaca EDF/hypnogram dan ringkasan EDA
├── src/                      # kode inti; diisi bertahap setelah alurnya terbukti di notebook
├── tests/                    # pengujian (belum dipakai)
├── requirements.txt          # dependensi langsung, lintas platform (tanpa PyTorch)
├── requirements-lock.txt     # snapshot persis lingkungan pengembang (Windows + CUDA 12.6)
└── README.md
```

Pustaka pembaca EDF di `scripts/edfx_eda.py` (`read_header`, `read_signals`, `read_annotations`) ditulis sendiri karena MNE mengubah EMG 1 Hz menjadi 100 Hz secara otomatis, sedangkan proyek ini mempertahankan EMG pada 1 Hz (30 sampel per epoch).

---

## 7. Progress dan roadmap

### Roadmap

- [x] Fase 0: lingkungan dan struktur proyek
- [x] EDA: profil dataset, rata-rata waktu per tahap, visualisasi kondisi data
- [ ] **Fase 1: pipeline data (sedang dikerjakan)**
- [ ] Fase 2: baseline klasik
- [ ] Fase 3: model deep learning pada SC-20
- [ ] Fase 4: benchmark SC-78 (*subject-wise* cross-validation)
- [ ] Fase 5: ablasi
- [ ] Fase 6: uji eksternal ST dan laporan

### Rincian Fase 1 (`notebooks/02_preprocessing.ipynb`)

| Bagian | Isi | Status |
|---|---|---|
| A | Konfigurasi dan parameter tetap | Selesai |
| B | Inventaris: memasangkan 153 PSG dengan hypnogram | Selesai |
| C | Anatomi satu rekaman: header, kanal, rentang nilai | Selesai |
| D | Label per epoch (pemetaan AASM, pemotongan ke panjang sinyal, label −1) dan uji silang dengan EDA | Selesai |
| E | *Trimming* jendela tidur ±30 menit. Uji unit: 195.469 epoch berlabel | Berikutnya |
| F | Penyaringan *band-pass* 0,3 sampai 35 Hz (EEG dan EOG) | Belum |
| G | Penanganan EMG (lihat keputusan K4) | Belum |
| H | Pemotongan epoch dan normalisasi per rekaman (*Robust Scaling*, *clip* ±20) | Belum |
| I | Pemeriksaan mutu seluruh 153 rekaman | Belum |
| J | Penyimpanan cache per rekaman dan verifikasi muat ulang | Belum |

Keluaran akhir Fase 1 per rekaman: EEG 2 kanal (N × 2 × 3.000), EOG (N × 3.000), EMG (N × 30), label (0 sampai 4, dan −1 untuk epoch yang diabaikan), serta metadata (ID subjek, malam, nomor epoch).

---

## 8. Temuan sejauh ini

### Profil dataset (EDA)

| Besaran | Nilai |
|---|---|
| Total rekaman | 197 (SC 153, ST 44) |
| Total durasi | 3.849,0 jam ≈ 461.870 epoch |
| Rekaman SC | 78 subjek, usia 25 sampai 101 tahun, rata-rata 22,7 jam per rekaman |
| Epoch berlabel SC-78 | 414.961 sebelum *trimming*, 195.469 sesudahnya |

Distribusi kelas SC-78 sebelum dan sesudah *trimming* ±30 menit:

| Kelas | Sebelum | Sesudah |
|---|---|---|
| Wake | 68,8% | 33,7% |
| N1 | 5,2% | 11,0% |
| N2 | 16,7% | 35,4% |
| N3 | 3,1% | 6,7% |
| REM | 6,2% | 13,2% |

Proporsi N3 turun seiring usia: 13,5% pada usia 25 sampai 39 tahun menjadi 2,8% pada usia 80 tahun ke atas.

### Baseline klasik awal (GBDT dengan fitur buatan tangan, validasi *subject-wise*)

| Konfigurasi | Macro-F1 (%) |
|---|---|
| EEG Fpz-Cz saja | 59,6 |
| + EEG Pz-Oz | 65,3 |
| + EOG | 67,9 |
| + EMG | 69,1 |
| + konteks ±2 epoch | 73,7 |
| + pembobotan kelas | 74,4 |

Bukti kebocoran data: dengan konfigurasi yang sama, pembagian acak per epoch menghasilkan Macro-F1 81,0%, sedangkan pembagian per subjek menghasilkan 73,7%. Selisih ±7 poin itulah yang membuat validasi wajib berbasis subjek.

Berkas hasil ada di `results/ablation/`. Skrip penghasilnya sudah dihapus dari pohon kerja karena tahapnya belum dimulai (tersedia di riwayat git pada commit `6dfc9b9`). Eksperimen akan diulang pada model akhir.

---

## 9. Keputusan desain

Dicatat lengkap di [docs/catatan/KEPUTUSAN_DESAIN.md](docs/catatan/KEPUTUSAN_DESAIN.md).

| Kode | Keputusan | Status |
|---|---|---|
| K1 | Epoch *Unscored* dan *Movement time* tetap berada di urutan dengan label −1, dikecualikan dari *loss* dan metrik | Diputuskan |
| K2 | Fase 1 menyimpan dua kanal EEG; 1 atau 2 kanal diuji sebagai ablasi | Ditunda ke pemodelan |
| K3 | Eksperimen kebocoran data diulang pada model akhir | Ditunda ke pemodelan |
| K4 | Transformasi EMG (nilai negatif pada 119 dari 153 rekaman membuat `log` langsung tidak bisa dipakai) | Terbuka, diputuskan di Bagian G |

---

## 10. Dokumentasi

| Berkas | Isi |
|---|---|
| [docs/analisis/ANALISIS_PROJECT_SLEEP_STAGING.md](docs/analisis/ANALISIS_PROJECT_SLEEP_STAGING.md) | Analisis proyek secara menyeluruh |
| [docs/belajar/01_eda_penjelasan.md](docs/belajar/01_eda_penjelasan.md) | Penjelasan notebook EDA sel demi sel |
| [docs/belajar/02_preprocessing_penjelasan.md](docs/belajar/02_preprocessing_penjelasan.md) | Penjelasan notebook pre-processing (Bagian A sampai D) |
| [docs/catatan/KEPUTUSAN_DESAIN.md](docs/catatan/KEPUTUSAN_DESAIN.md) | Keputusan desain dan alasannya |
| [docs/catatan/CATATAN_PERUBAHAN_REF1.md](docs/catatan/CATATAN_PERUBAHAN_REF1.md) | Daftar perubahan untuk dokumen kerangka penelitian |

---

## 11. Panduan kolaborasi

**Alur kerja.**

1. Ambil perubahan terbaru sebelum mulai: `git pull`.
2. Kerjakan di cabang sendiri: `git switch -c feat/<topik-singkat>`.
3. Commit kecil dan bermakna, lalu buka *pull request* ke `master` agar dapat ditinjau.

**Format pesan commit** (bahasa Inggris, gaya Conventional Commits): `feat(preprocessing): add trimming section`, `fix(eda): ...`, `docs: ...`, `chore: ...`, `refactor: ...`.

**Aturan yang tidak boleh dilanggar.**

| Aturan | Alasan |
|---|---|
| Jangan menulis ke `data/raw/` | Sumber data harus tetap sama untuk semua anggota |
| Jangan meng-commit data, cache `.npz`, atau bobot model | Ukurannya besar dan sudah diabaikan oleh `.gitignore` |
| Pembagian *fold* selalu berdasarkan ID subjek | Satu subjek memiliki dua malam rekaman; pembagian per berkas menimbulkan kebocoran data |
| Jangan menaruh kunci atau kata sandi di repositori | Berkas `.env` diabaikan git; gunakan itu bila perlu |

**Notebook.**

- Notebook di-commit **beserta keluarannya** agar pembaca dapat melihat hasil tanpa menjalankan ulang. Sebelum commit, pilih *Restart Kernel and Run All* supaya keluarannya konsisten.
- Satu notebook sebaiknya dikerjakan satu orang pada satu waktu. Notebook adalah berkas JSON dan sulit digabung bila diedit bersamaan. Bila terjadi konflik, ambil satu versi lalu jalankan ulang seluruh sel.
- Fungsi yang sudah stabil dipindahkan ke `src/` setelah alurnya terbukti di notebook, bukan sebelumnya.

**Bahasa.** Dokumen dan penjelasan memakai bahasa Indonesia. Kode, komentar kode, dan pesan commit memakai bahasa Inggris.

**Akhir baris.** Berkas `.gitattributes` menyeragamkan akhir baris di Windows, macOS, dan Linux. Peringatan `LF will be replaced by CRLF` dari git tidak berbahaya.

---

## 12. Pemecahan masalah

| Gejala | Penyebab umum | Solusi |
|---|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Kebijakan eksekusi PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, atau panggil `.\.venv\Scripts\python.exe` langsung |
| `ModuleNotFoundError: No module named 'numpy'` (atau paket lain) | Kernel notebook bukan `.venv`, atau lingkungan belum aktif | Pilih kernel `.venv` (bagian 4.8). Di terminal, pastikan awalan `(.venv)` tampak |
| `ModuleNotFoundError: No module named 'edfx_eda'` atau `StopIteration` pada sel pertama | Notebook dibuka di luar folder repositori | Buka folder root repositori di VS Code, bukan hanya berkas notebook |
| `AssertionError` pada sel inventaris (jumlah pasangan bukan 153) | Dataset belum ada atau salah letak | Cocokkan dengan tata letak di bagian 4.7 (tanpa folder perantara) |
| `FileNotFoundError ... data/raw/...` | Dataset belum diunduh | Ikuti bagian 4.7 |
| `IndexError` saat menjalankan `edfx_eda.py` | Folder `sleep-telemetry` tidak ada | Unduh dataset **lengkap** (SC dan ST) |
| `torch.cuda.is_available()` mengembalikan `False` | Build PyTorch CPU terpasang, atau driver NVIDIA lama | Pasang ulang dengan jalur `cu126` (bagian 4.5) dan periksa `nvidia-smi` |
| `pip` gagal memasang versi yang diminta | Versi Python bukan 3.11 | Buat ulang `.venv` dengan Python 3.11 |
| `UnicodeEncodeError` di konsol Windows | Konsol memakai kode halaman lama | Jalankan dari terminal dengan `.venv` aktif (skrip sudah memaksa keluaran UTF-8). Hindari ekstensi Code Runner yang memakai Python global |
| Kernel `.venv` tidak muncul di VS Code | VS Code belum memindai lingkungan baru | `Ctrl+Shift+P` lalu `Python: Select Interpreter`, atau muat ulang jendela |
| `sha256sum` melaporkan `FAILED` | Berkas rusak saat unduh | Unduh ulang berkas yang bersangkutan |
| `git status` menunjukkan `results/eda/...` berubah setelah menjalankan skrip | Data yang dipakai berbeda dari data proyek | Verifikasi data (bagian 4.7), lalu jalankan ulang |

Jika masalah belum teratasi, kirim ke grup: sistem operasi Anda, keluaran `python --version`, dan pesan galat lengkapnya.

---

## 13. Dataset, lisensi, dan sitasi

**Dataset.** [Sleep-EDF Database Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/), PhysioNet, DOI [10.13026/C2X676](https://doi.org/10.13026/C2X676). Lisensi dataset: *Open Data Commons Attribution License v1.0*. Dataset tidak disertakan di repositori ini; gunakan sesuai ketentuan lisensi tersebut dan kutip sumbernya.

**Sitasi yang diminta penyedia dataset:**

> B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberye. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. *IEEE-BME* 47(9):1185-1194 (2000).

Sitasi standar PhysioNet tercantum di halaman dataset.

**Lisensi kode.** Belum ditentukan oleh pemilik repositori. Sampai ada berkas lisensi, kode ini hanya untuk keperluan akademik proyek ini.

---

## 14. Penulis

Kelompok 8, mata kuliah *Deep Learning* (AI706), Informatika, Universitas Multimedia Nusantara. Pemilik repositori: Jap Feodrian Calvin (NIM 00000083500).
