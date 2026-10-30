# Automated Sleep Stage Classification using Multimodal Hybrid Deep Learning

Proyek akhir *Deep Learning* (Informatika, Universitas Multimedia Nusantara) untuk klasifikasi lima tahap tidur AASM (Wake, N1, N2, N3, REM) pada dataset PhysioNet Sleep-EDF Expanded. Model yang diusulkan adalah Multi-Branch 1D-CNN (EEG, EOG, EMG) yang digabung dengan 2-layer Bi-LSTM dan dilatih dengan Focal Loss. Evaluasi memakai *subject-wise* GroupKFold dengan metrik Macro-F1 dan Cohen's Kappa.

## Struktur Repositori

```
.
├── configs/            # konfigurasi percobaan (YAML), satu file per eksperimen
├── data/
│   ├── raw/sleep-edfx/ # data PhysioNet (tidak di-commit, lihat bagian Data)
│   └── processed/      # cache .npz hasil preprocessing (tidak di-commit)
├── docs/
│   ├── analisis/       # dokumen analisis project
│   └── catatan/        # catatan perubahan dan komentar dokumen
├── notebooks/          # EDA dan visualisasi hasil
├── results/
│   ├── eda/            # ringkasan EDA
│   ├── ablation/       # keluaran ablasi baseline
│   ├── figures/        # grafik
│   ├── tables/         # tabel hasil
│   ├── logs/           # log training
│   └── checkpoints/    # bobot model (tidak di-commit)
├── scripts/            # skrip EDA, ekstraksi fitur, dan ablasi baseline
├── src/sleepstaging/   # kode inti (data, model, loss, training, evaluasi)
├── tests/              # pengujian (jumlah epoch, tidak ada subjek yang tumpang tindih)
├── requirements.txt
└── README.md
```

## Setup

Persyaratan: Python 3.11 dan GPU NVIDIA (dikembangkan pada RTX 3050 6 GB).

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

Verifikasi GPU:

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
```

## Data

Dataset: [Sleep-EDF Database Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) (PhysioNet). Unduh lalu letakkan isinya di `data/raw/sleep-edfx/` sehingga terdapat folder `sleep-cassette/`, `sleep-telemetry/`, serta berkas `RECORDS`, `SC-subjects.xls`, dan `ST-subjects.xls`. Data tidak disertakan di repositori ini. Gunakan dataset sesuai ketentuan lisensi dan kutip sumbernya.

Pembagian penggunaan:

- **SC (Sleep-Cassette, 153 rekaman, 78 subjek):** data utama pelatihan dan evaluasi.
- **ST (Sleep-Telemetry, 44 rekaman, 22 subjek):** hanya uji generalisasi *out-of-distribution* karena adanya pengaruh Temazepam dan perbedaan perangkat EMG.

## Menjalankan EDA

Dari root repositori:

```bash
.venv/Scripts/python.exe -P scripts/edfx_eda.py data/raw/sleep-edfx results/eda
```

## Status

- [x] Fase 0: lingkungan dan struktur proyek
- [ ] Fase 1: pipeline data (cache `.npz`, uji jumlah epoch)
- [ ] Fase 2: baseline klasik
- [ ] Fase 3: model deep learning pada SC-20
- [ ] Fase 4: benchmark SC-78 (subject-wise cross-validation)
- [ ] Fase 5: ablasi
- [ ] Fase 6: uji eksternal ST dan laporan
