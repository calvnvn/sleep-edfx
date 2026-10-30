# Analisis Project: Automated Sleep Stage Classification using Multimodal Hybrid Deep Learning

> **Penulis project:** Jap Feodrian Calvin (00000083500) — Informatika UMN
> **Dataset:** PhysioNet Sleep-EDF Expanded v1.0.0 (lokal: `D:\sleep-edfx`)
> **Disusun:** 26 Oktober 2026 — berdasarkan (Ref-3) *Literature Review Map* UTS AI706, (Ref-2) spreadsheet *Kelompok8_ResearchGap_Assignment*, (Ref-1) dokumen *Klasifikasi Tahap Tidur Otomatis Berbasis Deep Learning*, ditambah **EDA langsung pada 197 rekaman EDF** dan **eksperimen baseline ablation** yang saya jalankan di mesin Anda.

---

## 0. Ringkasan Eksekutif (TL;DR)

**Penilaian umum:** Rancangan project (*Multi-Branch 1D-CNN + 2-layer Bi-LSTM + Focal Loss + Subject-wise GroupKFold*) **sudah berada di jalur yang benar** dan sejalan dengan praktik terbaik literatur (DeepSleepNet → TinySleepNet → MMASleepNet). Masalahnya bukan di arah besar, tetapi di **detail implementasi, konsistensi antar-dokumen, dan klaim kebaruan**.

**7 temuan terpenting:**

| # | Temuan | Dampak | Status |
|---|---|---|---|
| 1 | Kanal **EMG di SC bukan sinyal mentah**, melainkan *envelope* RMS 1 Hz (prefilter `HP:16Hz Rectification LP:0.7Hz`). Rencana *bandpass 0.5–30 Hz* + *upsampling* ke 100 Hz pada kanal ini **akan menghapus informasi tonus otot** (level DC = informasi atonia). | **Kritis** — cabang EMG bisa jadi tidak berguna | Harus diperbaiki |
| 2 | Klaim "**N1 < 5%**" dan "**Wake ≈ 65%**" hanya berlaku pada data **sebelum** *wake trimming*. Setelah trimming (yang Anda rencanakan), N1 = **11,0%** (SC-78) / **6,6%** (SC-20), Wake = 33,7% / 19,6%. | Narasi & justifikasi Focal Loss perlu dikoreksi | Koreksi angka |
| 3 | Angka "**483.428 epoch**" tidak konsisten dengan 3.849 jam (= 461.884 epoch). Angka 483k berasal dari panjang anotasi hypnogram, yang **melebihi panjang sinyal PSG pada 150/153 rekaman SC**. | Bug *alignment* potensial | Koreksi + crop label ke panjang sinyal |
| 4 | **Inkonsistensi metodologi antar-dokumen**: Ref-2 mengusulkan mengadopsi *Wavelet db4* (Zhao) dan *VMD+FFT* (Zhu), sedangkan Ref-1 menolak keduanya karena "komputasi tinggi". Ref-2 juga bergantian menulis "LSTM" vs "Bi-LSTM". Ref-3 memakai 197 rekaman/100 subjek, Ref-1 menyatakan SC & ST **tidak boleh** digabung. | Reviewer/dosen akan menangkap kontradiksi ini | Tetapkan satu desain final |
| 5 | **Klaim research gap terlalu kuat**: CoSleepNet (2023) sudah memakai CNN-LSTM + **Focal Loss** + **subject-wise k-fold** (EEG-EOG), MMASleepNet (2022) sudah memakai **EEG+EOG+EMG** + subject-grouped k-fold. Yang benar-benar baru hanyalah *kombinasi* ketiganya + analisis ablation yang terkontrol. | Risiko dinilai "tidak novel" | Reframe kontribusi → *controlled ablation study* |
| 6 | **DOI CoSleepNet di Ref-2 salah** (`10.1109/TNSRE.2021.3076234` adalah DOI AttnSleep/Eldele). DOI yang benar: **`10.1016/j.bspc.2022.104299`** (Biomed. Signal Process. Control, vol. 80, 104299, 2023). | Sitasi salah | Perbaiki |
| 7 | **SC-20 hanya berisi subjek usia 25–34 tahun**, sedangkan SC-78 berusia 25–101 tahun. Proporsi N3 turun dari 13,5% (25–40 th) ke 2,8% (≥80 th), dan **40 dari 153 rekaman hampir tanpa N3 (<1%)**. Hasil SC-20 vs SC-78 tidak bisa dibandingkan secara langsung. | Interpretasi hasil & per-subject metric | Laporkan stratifikasi usia |

**5 aksi prioritas berikutnya** (detail di §9–§10):
1. Bekukan desain final: **SC-78 sebagai benchmark utama (10-fold subject-wise), SC-20 untuk iterasi cepat (20-fold / LOSO-subject), ST sebagai *out-of-distribution test*** (bukan digabung ke training).
2. Perbaiki preprocessing EMG: **jangan di-bandpass, jangan di-upsample**; pakai 30 sampel/epoch dinormalisasi robust per rekaman (median/IQR, tanpa log) dengan cabang ringan.
3. Bangun *pipeline* data yang benar: crop label ke panjang sinyal, trimming 30 menit, normalisasi **per-rekaman (per malam)** yang *robust*, simpan per-rekaman ke `.npz` (float16).
4. Jalankan **matriks ablation** (modalitas × loss × konteks temporal × skema validasi) — ini yang menjadi kontribusi ilmiah utama.
5. Laporkan dengan standar benchmark: ACC, MF1, κ, F1 per-kelas, confusion matrix gabungan semua fold, mean±SD per-fold, dan uji statistik antar-konfigurasi.

---

## 1. Pemahaman Konteks Project

### 1.1 Peta tiga dokumen referensi

| Dokumen | Peran | Isi inti | Catatan kualitas |
|---|---|---|---|
| **Ref-3** — *Literature Review Map* (UTS AI706, 16 Okt 2026) | Landasan akademik / sitasi terverifikasi | RQ: *bagaimana arsitektur hybrid DL pada Sleep-EDF berbeda dalam desain input, penanganan imbalance, dan protokol validasi, serta pengaruhnya ke MF1 & κ*. 4 tema (T1 arsitektur, T2 input, T3 imbalance/N1, T4 validasi). 7 sumber final (DeepSleepNet, Zhu 2020, AttnSleep, MMASleepNet, Liu 2022, Fu 2021, Loh 2020) + log prompt & validasi sitasi. | **Paling kuat secara metodologis** — ada verifikasi DOI dan cross-check angka ke PDF. Research gap dirumuskan hati-hati ("dalam tujuh sumber yang ditinjau"). |
| **Ref-2** — Spreadsheet Research Gap (Kelompok 8) | Tabel 10 paper + gap + kesimpulan | Li 2022 (EEGSNet), Liu 2024 (review), Zhao 2022 (1D CNN-LSTM), SalientSleepNet 2021, Zhu 2025 (VMD+CNN-BiLSTM), Duan 2021, STQS 2021, Satapathy 2024, TSEDSleepNet 2024, CoSleepNet 2023. | Isi bagus, tetapi **kolom "Metode yang akan digunakan" saling bertentangan** (Wavelet, VMD+FFT, auxiliary classifier, GELU+residual, LSTM vs Bi-LSTM). DOI CoSleepNet salah. |
| **Ref-1** — Dokumen penjelasan project (hasil *scraping* AI) | Penjelasan konsep, pipeline, arsitektur, EDA | 7 poin pipeline (EDA → preprocessing → feature engineering → arsitektur → evaluasi → pipeline → gap), penjelasan fisiologi tidur, glossary, justifikasi 1D-CNN vs alternatif, Focal Loss, GroupKFold, profil dataset SC vs ST. | Sangat membantu sebagai materi presentasi, tetapi **beberapa angka & klaim teknis keliru** (lihat §6.3) — wajar karena hasil generatif AI, perlu diverifikasi. |

### 1.2 Inti masalah yang diselesaikan

```
Sinyal PSG semalam ──► potong per 30 detik (epoch, 3000 titik @100 Hz)
                    ──► klasifikasi tiap epoch ke {W, N1, N2, N3, REM} (AASM)
Tantangan: (1) N1 langka & ambigu   (2) W vs REM mirip di EEG → butuh EOG/EMG
           (3) ketergantungan temporal antar-epoch   (4) generalisasi ke subjek baru
```

### 1.3 Rancangan yang diusulkan (konsolidasi dari ketiga dokumen)

```
Input per epoch: EEG Fpz-Cz (3000) | EOG horizontal (3000) | EMG submental
        │                │                       │
   1D-CNN branch    1D-CNN branch           1D-CNN branch
     (64-d)            (64-d)                  (64-d)
        └───────── concat → 192-d feature vector ─────────┘
                              │   (urutan 10 epoch)
                  2-layer Bi-LSTM (128 + 128)
                              │
                  Softmax 5 kelas per epoch
Loss: Focal Loss (α, γ) | Validasi: GroupKFold by Subject ID
Metrik: Macro-F1 (utama), Cohen's κ, ACC, confusion matrix
```

Rancangan ini **masuk akal dan sebanding dengan model-model benchmark** (TinySleepNet ~1,3 M parameter; rancangan Anda ~0,6–1,2 M). Rumusan masalah riset di Ref-3 juga sudah tepat dan terukur.

---

## 2. Analisis Dataset Sleep-EDF Expanded (hasil EDA langsung)

> Semua angka di bagian ini **dihitung langsung** dari file EDF di `D:\sleep-edfx` menggunakan parser EDF/EDF+ yang saya tulis (tanpa MNE, karena belum terpasang). Skrip: `scripts/edfx_eda.py`. Output mentah: `results/eda/eda_summary.txt`.

### 2.1 Struktur folder & file

```
data/raw/sleep-edfx/                 (8,2 GB, 400 file)
├── RECORDS            197 baris (153 SC + 44 ST)
├── RECORDS-v1         61 baris  (subset versi lama)
├── SC-subjects.xls    metadata SC (usia, jenis kelamin, lights-off)
├── ST-subjects.xls    metadata ST (plasebo vs temazepam per malam)
├── SHA256SUMS.txt     checksum
├── sleep-cassette/    306 file = 153 PSG + 153 Hypnogram
└── sleep-telemetry/    88 file = 44 PSG + 44 Hypnogram
```

Penamaan: `SC4ssN` → `ss` = ID subjek (00–82), `N` = malam ke-1/2. Contoh `SC4001E0-PSG.edf` + `SC4001EC-Hypnogram.edf` (huruf ke-8 = kode *scorer*, berbeda antar-file → **jangan mencocokkan nama file secara persis, cocokkan 7 karakter pertama**).

### 2.2 Layout kanal (diverifikasi dari header EDF)

| Sub-set | Kanal | fs | Prefilter (header EDF) |
|---|---|---|---|
| **SC** (153 rekaman) | EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal | 100 Hz | `HP:0.5Hz LP:100Hz [enhanced cassette BW]` |
| | Resp oro-nasal | 1 Hz | `HP:0.03Hz LP:0.9Hz` |
| | **EMG submental** | **1 Hz** | **`HP:16Hz Rectification LP:0.7Hz`** → *envelope* µV-RMS |
| | Temp rectal, Event marker | 1 Hz | — |
| **ST** (44 rekaman) | EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal | 100 Hz | `LP:800Hz HP:0.03Hz` |
| | **EMG submental** | **100 Hz** | **`LP:800Hz HP:0.03Hz`** → sinyal "mentah" tetapi di-*sampling* 100 Hz |
| | Marker | 10 Hz | — |

**Implikasi penting:**
- EMG SC sudah merupakan *amplitude envelope*; nilai absolutnya (level tonus) adalah informasinya. **Bandpass 0,5–30 Hz akan membuang komponen < 0,5 Hz = membuang level tonus** — persis informasi atonia REM yang Anda butuhkan.
- EMG ST mengandung energi otot (10–500 Hz) yang di-*sampling* 100 Hz → **ter-*aliasing***. Tidak setara dengan EMG SC. Bila ST dipakai sebagai *test set*, EMG ST perlu diubah menjadi envelope 1 Hz (rectify → low-pass 0,7 Hz → resample 1 Hz) agar meniru SC, dan hasilnya tetap harus dibaca dengan hati-hati.
- EEG SC sudah *high-pass* 0,5 Hz secara hardware; EEG ST hanya 0,03 Hz → normalisasi & filter yang konsisten penting bila menguji lintas-sub-set.

### 2.3 Profil subjek & durasi

| | SC | ST |
|---|---|---|
| Rekaman / subjek | 153 / 78 (75 subjek 2 malam; **SC13, SC36, SC52 hanya 1 malam**) | 44 / 22 (semua 2 malam) |
| Usia | 25–101 th (median 57) | 18–78 th (median 35) |
| Jenis kelamin | 41 F / 37 M | 15 F / 7 M |
| Durasi per rekaman | rata-rata **22,7 jam** (17–24 jam), mulai ±14:50–17:00 | rata-rata **8,6 jam** (7,5–10,7 jam), mulai ±23:00 |
| Konteks | Subjek sehat, rekaman ambulatori di rumah (studi usia) | Subjek dengan *mild difficulty falling asleep*, studi efek **temazepam** vs plasebo (in-lab) |
| Total durasi | 3.470,2 jam | 378,8 jam |

**Total semua: 3.849,0 jam = 461.884 epoch 30 detik** (bukan 483.428 — lihat §6.3).

**SC-20 (benchmark "Sleep-EDF-20") = subjek SC00–SC19 = 39 rekaman, usia hanya 25–34 tahun.** Ini fakta penting yang jarang disadari: SC-20 adalah populasi *muda*; SC-78 mencakup lansia.

### 2.4 Distribusi kelas (5 kelas AASM: S3+S4 → N3, MT & ? dibuang)

| Subset | Kondisi | Epoch berlabel 5-kelas | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| **SC-78** (153 rek.) | Penuh (24 jam) | 414.961 | **68,8%** | 5,2% | 16,7% | 3,1% | 6,2% |
| **SC-78** | **Trim ±30 menit** | **195.469** | 33,7% | **11,0%** | 35,4% | 6,7% | 13,2% |
| **SC-20** (39 rek.) | Penuh | 106.414 | 68,0% | 2,6% | 16,7% | 5,4% | 7,3% |
| **SC-20** | **Trim ±30 menit** | **42.307** | 19,6% | **6,6%** | 42,1% | 13,5% | 18,2% |
| **ST** (44 rek.) | Penuh | 42.691 | 10,4% | 8,6% | 46,5% | 15,0% | 19,6% |
| **ST** | Trim ±30 menit | 42.467 | 9,9% | 8,6% | 46,7% | 15,1% | 19,7% |

Jumlah absolut setelah trim cocok persis dengan literatur (SC-78: N1 = 21.522, N2 = 69.132, N3 = 13.039, REM = 25.835; SC-20: W = 8.284, N1 = 2.804, N2 = 17.799, N3 = 5.703, REM = 7.717), sehingga pipeline label sudah tervalidasi. Variasi jumlah W antar-paper (65.9k–69.8k di SC-78) berasal dari perbedaan aturan trimming → **tulis aturan trimming secara eksplisit di laporan**.

**Kesimpulan imbalance yang benar:** setelah trimming, kelas minoritas sesungguhnya adalah **N3 di SC-78 (6,7%)** dan **N1 di SC-20 (6,6%)**. N1 tetap kelas *tersulit* (ambigu, bukan hanya langka). Rasio mayoritas:minoritas ≈ 5:1 — "imbalance moderat", bukan "ekstrem".

### 2.5 Efek usia (SC, setelah trim)

| Usia | Rekaman | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|
| 25–39 | 39 | 19,6% | 6,6% | 42,1% | **13,5%** | 18,2% |
| 40–59 | 41 | 33,2% | 10,3% | 37,7% | 5,5% | 13,4% |
| 60–79 | 41 | 31,2% | **13,7%** | 35,9% | 6,2% | 13,0% |
| ≥80 | 32 | **48,0%** | 12,6% | 27,3% | **2,8%** | 9,3% |

Per-rekaman: N1 berkisar 1,4%–42,1%; **40 rekaman SC memiliki N3 < 1%**. Konsekuensi: (a) F1 N3 per-subjek tidak terdefinisi untuk banyak lansia → hitung metrik dari prediksi gabungan semua fold; (b) model yang bagus di SC-20 belum tentu bagus di lansia → laporkan hasil terstratifikasi usia (ini bisa menjadi *nilai tambah* laporan Anda).

### 2.6 Dinamika transisi (SC-78, trim) — justifikasi Bi-LSTM

`P(stage berikutnya | stage sekarang)`:

| dari \ ke | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| **W** | **0,946** | 0,050 | 0,002 | 0,000 | 0,001 |
| **N1** | 0,089 | **0,724** | 0,158 | 0,000 | 0,028 |
| **N2** | 0,015 | 0,027 | **0,908** | 0,038 | 0,012 |
| **N3** | 0,010 | 0,005 | 0,187 | **0,797** | 0,001 |
| **REM** | 0,018 | 0,027 | 0,013 | 0,000 | **0,942** |

- 89,8% epoch memiliki label sama dengan epoch berikutnya → konteks temporal sangat informatif (inilah alasan kuantitatif memakai Bi-LSTM).
- **N1 paling "tidak stabil"** (hanya 72,4% bertahan) — N1 adalah *stage transisi*; kesalahan N1 terutama terjadi di batas W↔N1↔N2. Ini sejalan dengan Rosenberg & Van Hout (2013): kesepakatan antar-scorer manusia untuk N1 hanya **63,0%** (rata-rata keseluruhan 82,6%).
- Transisi yang praktis tidak pernah terjadi (W→N3, N3→REM, N1→N3) bisa dipakai sebagai *sanity check* prediksi (atau *post-processing* HMM/Viterbi sebagai eksperimen tambahan).
- Ada 6.864 epoch "terisolasi" (berbeda dari kedua tetangganya) — *smoothing* berlebihan akan merusak epoch ini, jadi Bi-LSTM lebih tepat daripada filter median sederhana.

### 2.7 Masalah kualitas data yang harus ditangani pipeline

| Isu | Temuan | Penanganan |
|---|---|---|
| Hypnogram lebih panjang dari PSG | 150/153 rekaman SC (anotasi melewati akhir sinyal hingga ~1,9 jam) | Crop label ke `floor(durasi_sinyal/30)` |
| Hypnogram lebih pendek dari PSG | 38/44 rekaman ST (sisa sinyal tanpa label) | Buang epoch tanpa label |
| Label `Sleep stage ?` | 1.335 epoch SC, 2.672 epoch ST (5,9%!) | Buang |
| `Movement time` | 128 SC, 83 ST | Buang (konvensi standar) |
| Subjek 1 malam | SC13, SC36, SC52 | Tidak masalah untuk GroupKFold, tapi fold tidak persis seimbang |
| Metadata *lights-off* | Ada di `SC-subjects.xls` | Alternatif trimming yang lebih "klinis" (opsional) |

---

## 3. Lanskap Literatur & Benchmark

### 3.1 Benchmark Sleep-EDF (subject-wise k-fold, angka dari paper masing-masing sebagaimana dikompilasi WaveSleepNet, 2024)

**Sleep-EDF-20 (Fpz-Cz, kecuali disebut lain)**

| Model | Tahun | ACC | MF1 | κ | F1-N1 |
|---|---|---|---|---|---|
| DeepSleepNet | 2017 | 82,0 | 76,9 | 0,76 | 46,6 |
| AttnSleep | 2021 | 84,4 | 78,1 | 0,79 | 42,6 |
| SleepEEGNet | 2019 | 84,3 | 79,7 | 0,79 | 52,2 |
| TinySleepNet | 2020 | 85,4 | 80,5 | 0,80 | 51,4 |
| XSleepNet | 2021 | 86,3 | 80,6 | 0,81 | — |
| **MMASleepNet (EEG+EOG+EMG)** | 2022 | — | **82,65** | **0,826** | **54,75** |

**Sleep-EDF-78**

| Model | ACC | MF1 | κ | F1-N1 |
|---|---|---|---|---|
| AttnSleep | 81,3 | 75,1 | 0,74 | 42,1 |
| SleepTransformer | 81,4 | 74,3 | 0,74 | 40,4 |
| SeqSleepNet (multi-ch) | 82,6 | 76,4 | 0,76 | — |
| TinySleepNet | 83,1 | 78,1 | 0,77 | 51,0 |
| XSleepNet | 83,9 | 77,9 | 0,78 | 49,9 |
| **MMASleepNet (EEG+EOG+EMG)** | — | **77,60** | **0,761** | **49,05** |
| CoSleepNet (EEG+EOG, Focal Loss)* | 87,11 | 79,83 | 0,818 | — |

\*CoSleepNet: hasil terbaik dari empat database, bukan murni SC-78 — jangan dibandingkan langsung.

**Target realistis untuk project Anda** (model ~1 M parameter, 1 GPU 6 GB):

| Dataset | MF1 | κ | F1-N1 | Interpretasi |
|---|---|---|---|---|
| SC-20 | 78–82 | 0,77–0,81 | 45–55 | ≥ 80 MF1 = setara SOTA |
| SC-78 | 75–79 | 0,74–0,78 | 45–52 | ≥ 78 MF1 = setara TinySleepNet |

Plafon praktis ditentukan oleh *inter-scorer agreement* (~82,6% overall, N1 ~63%) — **jangan menjanjikan F1-N1 > 60%**.

### 3.2 Posisi project dalam literatur (koreksi research gap)

| Studi | Multimodal EEG+EOG+**EMG** | Focal Loss | Subject-wise | Raw 1D + CNN + (Bi)LSTM |
|---|---|---|---|---|
| DeepSleepNet 2017 | ✗ | ✗ (oversampling) | ✓ | ✓ |
| TinySleepNet 2020 | ✗ | ✗ (weighted CE) | ✓ | ✓ |
| AttnSleep 2021 | ✗ | ✗ (weighted CE) | ✓ | ✗ (attention) |
| STQS 2021 | ✓ | ✗ | (SHHS) | ✓ |
| MMASleepNet 2022 | ✓ | ✗ (weighted CE) | ✓ (grouped) | ✗ (attention fusion) |
| CoSleepNet 2023 | ✗ (EEG+EOG) | **✓** | **✓** | ✓ (+DCT) |
| Satapathy 2024 | ✓ | ✗ | ✓ | ✗ (handcrafted ML) |
| **Project Anda** | **✓** | **✓** | **✓** | **✓** |

**Rumusan gap yang lebih aman dan jujur:**
> *"Belum ada studi yang secara terkontrol (protokol validasi seragam) mengisolasi kontribusi masing-masing modalitas (EEG, EOG, EMG), fungsi loss (CE, weighted CE, Focal), dan konteks temporal (CNN saja vs CNN+Bi-LSTM) terhadap F1-N1 dan Macro-F1 pada Sleep-EDF dengan validasi subject-wise, serta menguji generalisasinya ke populasi berbeda (lansia SC-78 dan subjek ST dengan temazepam)."*

Dengan framing ini, **kontribusi Anda adalah *ablation study* yang rapi + analisis generalisasi**, bukan "arsitektur baru" — jauh lebih mudah dipertahankan saat sidang/presentasi.

### 3.3 Referensi penting yang belum ada di ketiga dokumen

| Referensi | Kenapa penting |
|---|---|
| Supratak & Guo, **TinySleepNet**, EMBC 2020, doi:10.1109/EMBC44109.2020.9176741 | Baseline paling dekat dengan desain Anda (CNN + LSTM, ~1,3 M param, augmentasi sinyal & sekuens, weighted CE). **Wajib jadi baseline pembanding**. |
| Phan & Mikkelsen, *Automatic sleep staging of EEG signals: recent development, challenges, and future directions*, Physiol. Meas. 43(4) 04TR01, 2022 | Review otoritatif; membahas generalisasi, *inter-rater*, dan isu evaluasi. |
| Phan dkk., **SeqSleepNet** (2019), **XSleepNet** (2021), **SleepTransformer** (2022), **L-SeqSleepNet** (2023) | Pembanding sekuens panjang; L-SeqSleepNet menunjukkan manfaat konteks seluruh siklus tidur. |
| Perslev dkk., **U-Sleep**, npj Digit. Med. 2021 | Model lintas-dataset; konteks generalisasi. |
| Vallat & Walker, **YASA**, eLife 2021 | Baseline non-DL (fitur + LightGBM) yang kuat dan mudah dijalankan. |
| Efe & Özşen, **CoSleepNet**, BSPC 80:104299, 2023 | Paling dekat dengan klaim Focal Loss + subject-wise. |
| Rosenberg & Van Hout, JCSM 2013 | Plafon kesepakatan manusia (82,6%; N1 63%). |
| Thapa dkk., **SleepFM** (Nature Medicine 2026), SleepFM-2 (NeurIPS 2026) | Tren terbaru: *foundation model* multimodal PSG; bagus untuk bagian *future work*. |

---

## 4. Eksperimen Baseline Cepat (bukti empiris dari data Anda)

{{ABLATION_SECTION}}

---

## 5. Kelebihan Project

1. **Pertanyaan riset tajam & terukur** (Ref-3): empat tema (arsitektur, input, imbalance, validasi) langsung dipetakan ke metrik MF1/κ.
2. **Kesadaran tinggi terhadap data leakage** — banyak paper (Zhu 2020, Zhao 2022, Li 2022 EDF-8) jatuh di sini. Pilihan GroupKFold per subjek sudah benar, termasuk kesadaran bahwa 1 subjek = 2 malam.
3. **Pilihan arsitektur proporsional**: CNN raw-1D + Bi-LSTM adalah resep yang terbukti (DeepSleepNet/TinySleepNet) dan pas untuk GPU 6 GB. Tidak *over-engineered*.
4. **Pemahaman fisiologi yang baik** (Ref-1): peran EOG untuk SEM/REM dan EMG untuk atonia REM dijelaskan dengan benar dan menjadi justifikasi multimodal yang kuat.
5. **Metrik yang tepat**: Macro-F1 + Cohen's κ + confusion matrix, bukan sekadar akurasi.
6. **Proses literature review yang dapat diaudit** (Ref-3): log prompt, validasi sitasi, dan pengakuan eksplisit atas kesalahan ekstraksi AI — ini praktik yang sangat baik.
7. **Kesadaran SC vs ST** (Ref-1): memahami perbedaan hardware EMG dan efek temazepam — modal yang bagus untuk eksperimen generalisasi.

---

## 6. Kelemahan, Risiko & Koreksi

### 6.1 Kritis (akan merusak hasil bila tidak diperbaiki)

| # | Masalah | Mengapa bermasalah | Perbaikan |
|---|---|---|---|
| K1 | **Bandpass 0,5–30 Hz diterapkan ke semua kanal termasuk EMG** | EMG SC adalah envelope 1 Hz; HP 0,5 Hz membuang level tonus. Selain itu filter 30 Hz pada sinyal 1 Hz tidak bermakna (Nyquist 0,5 Hz). | Filter hanya EEG/EOG (mis. EEG 0,3–35 Hz, EOG 0,3–10 Hz atau biarkan). EMG: normalisasi robust per-rekaman (median/IQR), **tanpa log**: skala envelope berbeda antar-rekaman (median per rekaman −0,98 s/d 17,5) dan 55 dari 153 rekaman punya >1% nilai negatif. |
| K2 | **Upsampling EMG 1 Hz → 100 Hz** agar "sejajar" | Tidak menambah informasi (30 nilai diulang/diinterpolasi jadi 3000), memboroskan komputasi cabang CNN, dan bisa membuat CNN belajar artefak interpolasi. | Cabang EMG menerima **30 sampel/epoch** (atau 30 → Conv1d kecil / MLP). Multi-branch justru memungkinkan input berbeda panjang. |
| K3 | **Alignment label–sinyal** | Hypnogram SC melampaui sinyal pada 150/153 rekaman; tanpa crop, indeks epoch bisa bergeser/IndexError. | Hitung `n_epoch = floor(n_samples / 3000)`, potong label. Verifikasi dengan *assert* jumlah per kelas = tabel §2.4. |
| K4 | **Normalisasi "per subjek" sebelum trimming** | Bila z-score dihitung pada 24 jam rekaman SC (68% Wake siang hari + artefak), statistik normalisasi didominasi aktivitas siang. | Normalisasi **per-rekaman, setelah trimming**, pakai *robust scaling* (median/IQR) + *clipping* (mis. ±20 SD). |
| K5 | **Tuning/early stopping memakai test fold** | Bila *checkpoint* terbaik dipilih berdasarkan skor fold uji → optimistik (bocor secara metodologis). | Di setiap fold: sisihkan ~10% **subjek** dari training sebagai validasi (GroupShuffleSplit) untuk early stopping & pemilihan γ/α. |

### 6.2 Mayor (menurunkan kualitas ilmiah)

| # | Masalah | Perbaikan |
|---|---|---|
| M1 | **Kontradiksi desain antar-dokumen** (Wavelet/VMD diadopsi di Ref-2 tetapi ditolak di Ref-1; LSTM vs Bi-LSTM; auxiliary classifier & GELU/residual hanya muncul di Ref-2). | Tulis satu *Design Spec* final (lihat §8). Hal yang tidak dipakai dipindah ke "related work" atau "future work". |
| M2 | **SC+ST: digabung atau tidak?** Ref-3 menyebut 197 rekaman/100 subjek; Ref-1 menyatakan tidak boleh digabung. | Utama: **SC-78** (10-fold subject-wise). Iterasi cepat: **SC-20**. **ST hanya sebagai test OOD** (model dilatih di SC). |
| M3 | **Klaim novelty terlalu kuat** (lihat §3.2). | Reframe menjadi *controlled ablation & generalization study*. |
| M4 | **Penjelasan Focal Loss menyesatkan**: Focal Loss menurunkan bobot *sampel mudah*, bukan *kelas mayoritas*. N2 yang sulit tetap berbobot tinggi; N1 yang mudah tetap ditekan. Penyeimbangan kelas berasal dari **α**, bukan γ. | Bandingkan **CE vs weighted-CE vs Focal(γ ∈ {1, 2}) vs Focal+α**. Literatur sering menemukan weighted-CE ≈ Focal pada data ini — hasil apa pun tetap bernilai bila dilaporkan jujur. |
| M5 | **Panjang sekuens 10 epoch (5 menit)** mungkin terlalu pendek; siklus tidur ±90 menit. TinySleepNet/SeqSleepNet memakai 20–21 epoch; L-SeqSleepNet jauh lebih panjang. | Ablation `seq_len ∈ {1, 10, 20, 30}`; gunakan *many-to-many* (prediksi semua epoch di sekuens) + *sliding window* saat inferensi. |
| M6 | **Tidak ada augmentasi data** | Tambahkan augmentasi ringan: *random time-shift* (±0,5 s), *amplitude scaling* (0,8–1,2), *Gaussian noise*, dan *sequence shift* (cara TinySleepNet). Juga *modality dropout* (acak nolkan EOG/EMG) agar robust bila kanal hilang. |
| M7 | **Belum ada rencana uji statistik** | Laporkan mean±SD per fold; uji Wilcoxon *signed-rank* berpasangan per-fold/per-subjek antar-konfigurasi; laporkan *confidence interval* (bootstrap per subjek). |
| M8 | **DOI CoSleepNet salah** di Ref-2; Zhu 2025 (IEEE, 13 subjek) kualitas bukti rendah; Satapathy 2024 "91,08% subject-wise" dengan handcrafted ML — perlu dicek ulang protokolnya sebelum dikutip sebagai pembanding. | Perbaiki DOI → `10.1016/j.bspc.2022.104299`; beri label "bukti lemah" pada studi dengan protokol tidak jelas. |

### 6.3 Minor (koreksi faktual di dokumen)

| Klaim di dokumen | Fakta (dari data/literatur) |
|---|---|
| "483.428 epochs" | 461.884 epoch dari sinyal PSG (3.849,0 jam × 120). Angka 483k ≈ panjang anotasi hypnogram (484.111) yang melebihi sinyal. |
| "N1 < 5%" | Benar hanya sebelum trimming (5,2%). Setelah trimming: 11,0% (SC-78), 6,6% (SC-20), 8,6% (ST). |
| "Wake bias ≈ 65%" | Tepatnya 68,8% di SC-78 penuh; 33,7% setelah trimming. |
| "N2 > 50%" | N2 = 35,4% (SC-78 trim), 42,1% (SC-20 trim), 46,7% (ST). |
| "Model menebak N2 terus → akurasi ≈ 85%" | Menebak N2 semua hanya memberi akurasi ≈ 35–42%. Contoh ilustrasi perlu diganti. |
| "Fitur **spasial** dari 1D-CNN" | 1D-CNN per kanal mengekstrak fitur **temporal/morfologis/spektral**, bukan spasial (spasial = antar-elektroda). |
| "REM: gelombang Beta" | EEG REM = *low-amplitude mixed-frequency*, dominan theta + *sawtooth waves* (2–6 Hz). |
| "ST = 22 pasien insomnia" | Deskripsi PhysioNet: subjek dengan *mild difficulty falling asleep* (studi temazepam), bukan insomnia klinis terdiagnosis. |
| "Spektrogram 2D = *heavy overhead*" | STFT murah di GPU; SeqSleepNet/XSleepNet berbasis spektrogram tetap ringan. Alasan yang lebih kuat untuk raw-1D: *end-to-end* & kesederhanaan pipeline, bukan efisiensi. |
| "Transformer butuh jutaan data" | SleepTransformer dilatih di Sleep-EDF-78 dengan hasil kompetitif. Alasan yang lebih tepat: kompleksitas & sumber daya, bukan "pasti overfit". |
| "Sleep-EDF-20 berisi subjek sehat umum" | SC-20 = subjek **25–34 tahun** saja. |

---

## 7. Best Practice Checklist (Sleep Staging dengan DL)

**Data & preprocessing**
- [ ] Pilih kanal dari header EDF berdasarkan *label*, bukan indeks.
- [ ] Crop label ke panjang sinyal; buang `?` dan `MT`; S3+S4 → N3.
- [ ] Trimming W ±30 menit dari tidur pertama/terakhir (tulis aturannya).
- [ ] Filter hanya EEG/EOG; EMG envelope → log transform.
- [ ] Normalisasi per-rekaman, robust, setelah trimming; clip outlier.
- [ ] Simpan per-rekaman (`.npz`, float16) agar loading per fold efisien.
- [ ] *Unit test* jumlah epoch per kelas = tabel referensi.

**Validasi**
- [ ] GroupKFold berdasarkan **subject ID** (bukan record ID).
- [ ] Validation split di dalam training fold (berdasarkan subjek) untuk early stopping.
- [ ] Hasil akhir = prediksi dari semua fold digabung → hitung ACC/MF1/κ/per-class F1 + confusion matrix (konvensi literatur), plus mean±SD per fold.
- [ ] Satu eksperimen sengaja *epoch-wise* untuk **mengukur besarnya leakage** (kontribusi T4).

**Training**
- [ ] Seed tetap, `torch.backends.cudnn.deterministic` bila perlu, log semua konfigurasi.
- [ ] Mixed precision (AMP) untuk GPU 6 GB; batch = 15–32 sekuens × 20 epoch.
- [ ] Adam/AdamW lr 1e-3 → *cosine/plateau*, gradient clipping 1–5 untuk LSTM.
- [ ] Early stopping pada **val MF1** (bukan val loss).
- [ ] *Auxiliary loss* pada output CNN (per-epoch) membantu stabilitas (gagasan dari EEGSNet di Ref-2) — opsional, jadikan ablation.

**Pelaporan**
- [ ] Tabel perbandingan dengan baseline yang *Anda jalankan sendiri* di protokol yang sama (mis. TinySleepNet re-implementasi, YASA/GBDT fitur).
- [ ] Hipnogram prediksi vs ground truth untuk 2–3 subjek (visual kualitatif).
- [ ] Analisis kesalahan: confusion N1↔W, N1↔N2, N1↔REM; stratifikasi usia.
- [ ] Jumlah parameter, waktu training, waktu inferensi per malam.

---

## 8. Rekomendasi Desain Final (v2)

### 8.1 Arsitektur

```
EEG Fpz-Cz [1×3000] ─► CNN-EEG (dual-scale: kernel kecil 50/s6 & besar 400/s50, à la DeepSleepNet) ─► 128-d
EEG Pz-Oz  [1×3000] ─► (opsional, ablation) shared/separate CNN ──────────────────────────────────► 128-d
EOG        [1×3000] ─► CNN-EOG (lebih dangkal, kernel besar untuk SEM/REM) ───────────────────────► 64-d
EMG env    [1×30]   ─► robust-norm per rekaman → Conv1d(1,16,k=5) → GAP → Linear ──────────────────────────────────────► 32-d
                         │
             concat (224–352-d) → Modality gating (SE/attention bobot per modalitas, opsional)
                         → Dropout(0.5) → [Aux head: Linear→5, loss×0.5]
                         │  sekuens L = 20 epoch
             2-layer Bi-LSTM (hidden 128/arah, dropout 0.3)
                         │
             Linear → 5 kelas (many-to-many)
```

Perkiraan parameter: CNN-EEG ≈ 0,2 M, CNN-EOG ≈ 0,1 M, EMG < 0,01 M, Bi-LSTM 2-layer (input 256) ≈ 0,8 M → **total ≈ 1,1 M** (setara TinySleepNet).

### 8.2 Data & split

| Peran | Data | Protokol |
|---|---|---|
| Iterasi cepat / debugging | SC-20 (39 rek., 20 subjek) | 20-fold (1 subjek/fold) — sama dengan literatur |
| **Benchmark utama** | **SC-78** (153 rek., 78 subjek) | **10-fold subject-wise** |
| Uji generalisasi (OOD) | ST (44 rek.), latih di SC-78 | *Train once on all SC, test on ST*; EMG ST diubah ke envelope 1 Hz |
| Uji leakage (T4) | SC-20 | Epoch-wise shuffle k-fold vs subject-wise (konfigurasi identik) |

### 8.3 Matriks eksperimen (ablation) — inti kontribusi

| ID | Modalitas | Temporal | Loss | Validasi | Menjawab |
|---|---|---|---|---|---|
| E0 | Fitur + GBDT (YASA-like) | — | CE | subject | Baseline non-DL |
| E1 | EEG Fpz-Cz | CNN saja (L=1) | CE | subject | Baseline DL |
| E2 | EEG | CNN + Bi-LSTM (L=20) | CE | subject | T1: nilai konteks temporal |
| E3 | EEG + EOG | CNN + Bi-LSTM | CE | subject | T2 |
| E4 | EEG + EOG + EMG | CNN + Bi-LSTM | CE | subject | T2 |
| E5 | E4 | CNN + Bi-LSTM | weighted CE | subject | T3 |
| E6 | E4 | CNN + Bi-LSTM | Focal γ=2 (+α) | subject | T3 (**model usulan**) |
| E7 | E6 | — | — | **epoch-wise** | T4: besar leakage |
| E8 | E6 | L ∈ {10, 30} | Focal | subject | Sensitivitas panjang sekuens |
| E9 | E6 dilatih SC-78 | — | — | test ST | Generalisasi OOD |
| E10 | TinySleepNet re-impl. | — | weighted CE | subject | Pembanding SOTA pada protokol identik |

Jalankan E1–E7 di **SC-20** dulu (cepat, ±20 subjek), lalu konfigurasi terbaik + E9/E10 di **SC-78**.

---

## 9. Langkah Berikutnya (Next Steps)

| Prioritas | Aksi | Output | Estimasi |
|---|---|---|---|
| P0 | Setup environment: `torch` (CUDA), `mne`, `scikit-learn`, `pandas`, `matplotlib`, `pyyaml`, `tensorboard` (opsional `wandb`) | `requirements.txt` | 0,5 hari |
| P0 | Freeze *Design Spec* (dokumen 1 halaman: data, preprocessing, model, loss, split, metrik) + perbaiki angka/DOI di dokumen | `docs/design_spec.md` | 0,5 hari |
| P1 | Preprocessing → `data/processed/SC4001.npz` (eeg, eog, emg_env, labels, subject_id, age) + unit test distribusi kelas | Dataset siap | 1–2 hari |
| P1 | Baseline E0 (fitur + GBDT, sudah ada prototipe di §4) dan E1 | Angka baseline | 1 hari |
| P2 | Implementasi model v2 + training loop + CV runner + logging | `src/` | 3–4 hari |
| P2 | Ablation E1–E8 di SC-20 | Tabel ablation | 3–4 hari (GPU) |
| P3 | SC-78 10-fold untuk konfigurasi terbaik + E10 TinySleepNet | Tabel utama | 3–5 hari (GPU) |
| P3 | E9 test OOD pada ST + stratifikasi usia | Analisis generalisasi | 1–2 hari |
| P4 | Analisis kesalahan, hipnogram, interpretabilitas (Grad-CAM 1D / bobot gating modalitas) | Gambar & insight | 2 hari |
| P4 | Penulisan laporan/paper + slide | Laporan akhir | 3–4 hari |

**Estimasi beban GPU (RTX 3050 6 GB):** SC-78 trim ≈ 195 ribu epoch × 3 kanal × 3000 sampel ≈ 3,5 GB (float16). Dengan model ~1 M parameter dan AMP, satu *pass* data ±1–3 menit → 10 fold × ~40 epoch training ≈ 8–15 jam total. SC-20 ±5× lebih cepat. RAM 20 GB cukup bila data dimuat per-rekaman (bukan semua sekaligus dalam float32).

---

## 10. Framework Pengerjaan ke Depan

### 10.1 Struktur repository yang disarankan

```
sleep-staging/
├── configs/                 # YAML per eksperimen (E1.yaml … E10.yaml)
├── data/
│   ├── raw/sleep-edfx/       # data PhysioNet (tidak di-commit)
│   └── processed/           # 1 file .npz per rekaman
├── src/
│   ├── data/
│   │   ├── edf_io.py        # baca EDF + hypnogram (MNE atau parser sendiri)
│   │   ├── preprocess.py    # trim, mapping label, filter, normalisasi
│   │   └── dataset.py       # SequenceDataset (L epoch berurutan per rekaman)
│   ├── models/
│   │   ├── branches.py      # CNN-EEG, CNN-EOG, EMG-head
│   │   └── sleepnet.py      # fusion + Bi-LSTM + heads
│   ├── losses.py            # CE, weighted CE, FocalLoss(alpha, gamma)
│   ├── train.py             # 1 fold: train/val/early-stop/checkpoint
│   ├── cv.py                # GroupKFold runner → simpan prediksi per fold
│   └── evaluate.py          # metrik gabungan, CM, per-subjek, uji statistik
├── notebooks/               # EDA & visualisasi hasil
├── results/                 # predictions_*.npz, metrics_*.json, figures
└── tests/                   # test_label_counts.py, test_no_subject_overlap.py
```

### 10.2 Potongan kode kunci

**Split anti-leakage + validasi dalam fold**

```python
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

outer = GroupKFold(n_splits=10)
for fold, (trval_idx, test_idx) in enumerate(outer.split(rec_ids, groups=subject_ids)):
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=fold)
    tr, va = next(inner.split(trval_idx, groups=subject_ids[trval_idx]))
    train_recs, val_recs, test_recs = rec_ids[trval_idx[tr]], rec_ids[trval_idx[va]], rec_ids[test_idx]
    assert not set(subject_ids[trval_idx]) & set(subject_ids[test_idx])  # tidak ada subjek bocor
```

**Focal Loss (multi-kelas, dengan α per kelas)**

```python
import torch, torch.nn as nn, torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma

    def forward(self, logits, target):          # logits (N, C), target (N,)
        logp = F.log_softmax(logits, dim=-1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logp_t.exp()) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()

# α yang disarankan: inverse-sqrt frekuensi kelas di training fold, dinormalisasi rata-rata = 1
```

**Preprocessing EMG yang benar (SC)**

```python
emg = sig["EMG submental"]                       # 1 Hz envelope; skala/offset berbeda antar-rekaman, ada nilai negatif
emg = emg[:n_epochs * 30].reshape(n_epochs, 30)  # 30 sampel per epoch — TANPA bandpass/upsample/log
emg = emg[kept_epochs]                           # setelah trimming ±30 menit & buang label ?/MT
emg = (emg - np.median(emg)) / (iqr(emg) + 1e-6) # robust normalisasi per-rekaman (median/IQR)
emg = np.clip(emg, -10, 10)                      # batasi outlier
```

### 10.3 Alur kerja eksperimen

```
[Design Spec beku] → [Preprocess + unit test] → [Baseline E0/E1] → [Ablation SC-20]
        → [Pilih konfigurasi terbaik di VAL, bukan TEST] → [SC-78 10-fold] → [Test OOD ST]
        → [Error analysis + stratifikasi usia] → [Laporan]
```

Aturan emas: **setiap angka di laporan harus dapat direproduksi dari satu file config + satu seed.**

---

## 11. Skills & Tools yang Cocok untuk Analisis Deep Learning

### 11.1 Plugin/skill Claude yang relevan (tersedia di katalog Anthropic Directory)

| Plugin | Skill di dalamnya | Kegunaan untuk project ini |
|---|---|---|
| **Probabl Skills** (scikit-learn/Probabl) | `audit-ml-pipeline`, `evaluate-ml-pipeline`, `iterate-ml-experiment`, `smoke-test-ml-pipeline`, `test-ml-pipeline`, `explore-ml-data`, `organize-ml-workspace` | **Paling relevan**: audit leakage & protokol CV, struktur workspace eksperimen, iterasi ablation, smoke test pipeline sebelum training panjang. |
| **Data** (Anthropic) | `explore-data`, `statistical-analysis`, `create-viz`, `validate-data` | EDA, uji statistik antar-konfigurasi (Wilcoxon, bootstrap CI), visualisasi hasil. |
| **read-paper** | `read-paper` | Membedah paper (AttnSleep, TinySleepNet, CoSleepNet) secara terstruktur: kontribusi, metode, eksperimen. |
| **learn** | `teach`, `explain`, `why`, `quiz` | Mode *pair-programming* yang memastikan Anda paham setiap baris kode — cocok untuk persiapan sidang/presentasi. |

Skill bawaan yang sudah aktif di sesi ini dan berguna: **`dataviz`** (grafik hasil yang konsisten), **`xlsx`** (memperbaiki spreadsheet research gap), **`pdf`** (membaca paper PDF), **`docx` / `pptx`** (laporan akhir & slide).

### 11.2 Library Python yang direkomendasikan

| Library | Fungsi |
|---|---|
| `mne` | Membaca EDF/EDF+ & anotasi (`mne.io.read_raw_edf`, `mne.read_annotations`), filtering |
| `torch` (+ CUDA 12.x) | Model & training; AMP untuk GPU 6 GB |
| `scikit-learn` | `GroupKFold`, `GroupShuffleSplit`, metrik (`f1_score`, `cohen_kappa_score`) |
| `braindecode` | Referensi implementasi model EEG & loader Sleep-EDF (berguna untuk cross-check) |
| `yasa` | Baseline sleep staging berbasis fitur + hypnogram plotting |
| `tensorboard` / `wandb` / `mlflow` | Tracking eksperimen |
| `hydra-core` / `pyyaml` | Manajemen konfigurasi eksperimen |
| `scipy.stats` | Wilcoxon, bootstrap |

---

## 12. Lampiran

### 12.1 File analisis yang dibuat

| File | Isi |
|---|---|
| `scripts/edfx_eda.py` | Parser EDF/EDF+ (tanpa MNE) + EDA lengkap (kanal, durasi, distribusi kelas, usia, transisi) |
| `edfx_features.py`, `edfx_ablation.py`, `edfx_ablation_c.py` | Ekstraksi 26 fitur per-epoch dan ablasi GBDT (modalitas, konteks, bobot kelas, epoch-wise vs subject-wise). Sudah dihapus dari folder kerja karena belum dimulai bersama; tersedia di riwayat Git (commit `6dfc9b9`, folder `scripts/`) |
| `results/eda/eda_summary.txt` | Output EDA mentah |
| `results/ablation/ablation_results.txt`, `ablation_c.txt` | Output ablation mentah |

Menjalankan ulang (dari root repositori):

```bash
python scripts/edfx_eda.py data/raw/sleep-edfx results/eda
```

### 12.2 Sumber yang diverifikasi selama analisis

- PhysioNet Sleep-EDF Expanded v1.0.0 — https://physionet.org/content/sleep-edfx/1.0.0/
- CoSleepNet (DOI benar) — https://gcris.ktun.edu.tr/handle/20.500.13091/3160 (doi:10.1016/j.bspc.2022.104299)
- Tabel benchmark Sleep-EDF-20/78 & distribusi kelas — WaveSleepNet, https://arxiv.org/html/2404.15342
- Benchmark Sleep-EDF-78 (parameter model) — LightSleepX, https://arxiv.org/pdf/2609.19062
- Inter-scorer reliability AASM — https://pubmed.ncbi.nlm.nih.gov/23319910/
- Phan & Mikkelsen 2022 review — https://arxiv.org/abs/2111.08446
- TinySleepNet — https://repository.li.mahidol.ac.th/handle/123456789/59042
- SleepFM — https://pmc.ncbi.nlm.nih.gov/articles/PMC12920147/
- Satapathy 2024 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11071240/
- Duan 2021 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8531206/
