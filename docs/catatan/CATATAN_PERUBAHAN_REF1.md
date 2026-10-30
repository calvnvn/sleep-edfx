# Catatan Perubahan untuk Ref-1 — "Automated Sleep Stage Classification using Multimodal Hybrid Deep Learning" (7 poin)

> Dokumen yang direview: `KLASIFIKASI TAHAP TIDUR OTOMATIS BERBASIS DEEP LEARNING (1).md`
> Semua angka di bawah **dihitung langsung dari file EDF di `D:\sleep-edfx`** (skrip: `scripts/edfx_eda.py`), bukan dari literatur, kecuali ditandai.
> Prioritas: 🔴 **wajib** (salah fakta / merusak hasil) · 🟡 **sebaiknya** (kurang jelas / menyesatkan) · 🟢 **opsional** (kerapian)

## Ringkasan: 14 perubahan

| # | Bagian | Prioritas | Perubahan singkat |
|---|---|---|---|
| 1 | EDA – Dataset Profile | 🔴 | "22 pasien insomnia" → subjek dengan kesulitan tidur ringan; tegaskan sub-dataset yang dipakai |
| 2 | EDA – Data Volume | 🔴 | 483.428 epoch → ≈461.870 epoch |
| 3 | EDA – Critical Findings | 🔴 | "N1 < 5%", "N2 mendominasi", "Wake ≈ 65%" hanya benar *sebelum trimming*; "extreme" → moderat |
| 4 | Pre-processing #1 – Trimming | 🟡 | Definisikan aturan & beri angka sebelum/sesudah |
| 5 | Pre-processing #2 – Label | 🟡 | Tambah label `?`, crop label ke panjang sinyal, aturan pasangan file |
| 6 | Pre-processing #3 – Upsampling EMG | 🔴 | Hapus upsampling 1 Hz → 100 Hz |
| 7 | Pre-processing #4 – Bandpass & Z-score | 🔴 | Bandpass jangan diterapkan ke EMG; normalisasi per-rekaman setelah trimming |
| 8 | Feature Engineering | 🟡 | "spatial features" → fitur temporal/morfologi; EMG branch input berbeda |
| 9 | Arsitektur (teks + gambar) | 🔴 | Input "3 channel × 3000" salah untuk EMG; "Spatial" → "Epoch-level" |
| 10 | Evaluation – Focal Loss | 🟡 | Penjelasan Focal Loss keliru; tambah pembanding CE/weighted-CE |
| 11 | Evaluation – Validasi & Metrik | 🟡 | Tambah validation split dalam fold, definisi SC-20/SC-78, metrik tambahan, typo `\\%` |
| 12 | Pipeline (gambar 3) | 🟡 | Tambah validation set, uji OOD, crop label |
| 13 | Research Gap | 🟡 | Reframe klaim kebaruan; perbaiki sumber di spreadsheet |
| 14 | Gambar 1 & kerapian | 🟢 | Ganti screenshot EDFbrowser dengan grafik EDA; judul, istilah |

---

## 1. EDA – Dataset Profile 🔴

**Teks sekarang**
> Terdiri dari 197 *PSG records* dari 100 *unique subjects* (78 subjek sehat pada SC dan 22 **pasien insomnia** pada ST).

**Masalah**
- ST (Sleep-Telemetry) adalah studi efek **temazepam** pada subjek dengan *mild difficulty falling asleep* — bukan pasien insomnia terdiagnosis. Tiap subjek punya 1 malam plasebo + 1 malam temazepam (lihat `ST-subjects.xls`).
- Angka 197 benar sebagai isi database, **tetapi yang akan dipakai model hanya SC** (lihat #11). Dokumen sebaiknya menyebut ini di awal supaya tidak kontradiktif dengan bagian validasi.

**Ganti dengan**
> Dataset terdiri dari **197 rekaman PSG dari 100 subjek**: **Sleep-Cassette (SC)** — 153 rekaman dari 78 subjek sehat usia 25–101 tahun (rekaman ambulatori di rumah, 2 malam/subjek kecuali subjek 13, 36, 52), dan **Sleep-Telemetry (ST)** — 44 rekaman dari 22 subjek dengan kesulitan tidur ringan (studi efek temazepam, 1 malam plasebo + 1 malam obat). **Model dilatih dan dievaluasi pada SC**; ST hanya dipakai sebagai uji generalisasi karena pengaruh obat dan perbedaan hardware EMG.

**Tambahkan (baru):** SC-20 = subjek SC00–SC19 (39 rekaman, usia 25–34 th, identik dengan `RECORDS-v1`); SC-78 = seluruh SC (153 rekaman). Metadata usia/jenis kelamin/jam lampu-mati ada di `SC-subjects.xls` dan `ST-subjects.xls`.

---

## 2. EDA – Data Volume 🔴

**Teks sekarang**
> ≈ 3.849 hours … **483.428 epochs**

**Masalah:** 3.849,0 jam × 120 epoch/jam = **461.884**. Angka 483.428 mendekati panjang *anotasi hypnogram* (±484.111), yang **lebih panjang dari sinyal** pada 150 dari 153 rekaman SC (anotasi melewati akhir sinyal hingga ±1,9 jam). Dengan kata lain 483k mengandung epoch yang sebenarnya tidak punya sinyal.

**Ganti dengan**
> *Grand total duration* ≈ 3.849,0 jam (SC 3.470,2 jam + ST 378,8 jam) ≈ **461.870 epoch** 30 detik berdasarkan panjang sinyal. Setelah membuang label `Unscored (?)` (4.007 epoch) dan `Movement time` (211 epoch), tersisa **457.652 epoch berlabel 5 kelas** sebelum trimming dan **237.936** setelah trimming.

---

## 3. EDA – Critical Findings 🔴

**Teks sekarang**
> ketimpangan … *extreme* … **N2** mendominasi … **N1** … < 5% … *Wake bias* … ≈ 65% pada dataset SC

**Fakta dari data (5 kelas, epoch berlabel)**

| Subset | Kondisi | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|
| SC-78 | Sebelum trimming | **68,8%** | 5,2% | 16,7% | 3,1% | 6,2% |
| SC-78 | Setelah trimming ±30 mnt | 33,7% | **11,0%** | **35,4%** | **6,7%** | 13,2% |
| SC-20 | Sebelum trimming | 68,0% | 2,6% | 16,7% | 5,4% | 7,3% |
| SC-20 | Setelah trimming | 19,6% | **6,6%** | **42,1%** | 13,5% | 18,2% |
| ST | Setelah trimming | 9,9% | 8,6% | 46,7% | 15,1% | 19,7% |

**Masalah**
1. "N2 mendominasi" tidak benar sebelum trimming — yang dominan adalah **Wake (68,8%)**; N2 hanya 16,7%. N2 baru jadi mayoritas **setelah** trimming.
2. "N1 < 5%" tidak benar setelah trimming (11,0% di SC-78; 6,6% di SC-20).
3. Setelah trimming, kelas paling langka di SC-78 adalah **N3 (6,7%)**, bukan N1. Rasio mayoritas:minoritas ≈ 5:1 → **moderat**, bukan "extreme". N1 tetap yang *tersulit* (ambigu), bukan semata-mata paling sedikit.
4. Wake ≈ 65% mendekati benar (68,8%) tetapi hanya untuk SC sebelum trimming.

**Ganti dengan**
> Pada SC sebelum trimming, **Wake mendominasi (68,8%)** karena perekam berjalan ±22,7 jam per rekaman dan mencakup siang hari. Setelah *trimming* ±30 menit (SC-78), distribusi menjadi W 33,7%, **N1 11,0%**, **N2 35,4%**, **N3 6,7%**, REM 13,2% — ketimpangan **moderat** (rasio ≈ 5:1). N1 adalah kelas tersulit: hanya 72% epoch N1 diikuti N1 lagi (kelas paling tidak stabil), dan kesepakatan antar-scorer manusia untuk N1 hanya ±63% (Rosenberg & Van Hout, 2013). Usia sangat memengaruhi distribusi: N3 turun dari 13,5% (usia 25–39) menjadi 2,8% (usia ≥ 80), dan 40 dari 153 rekaman SC hampir tanpa N3 (< 1%).

---

## 4. Pre-processing #1 — 30-Minute Wake Trimming 🟡

**Teks sekarang:** "Memangkas ekor Wake epochs di luar rentang 30 menit sebelum sleep onset dan 30 menit sesudah sleep offset."

**Masalah:** benar secara konsep, tetapi jumlah epoch Wake berbeda antar-paper (SC-78: 65.9k di data saya vs 69.8k di literatur) karena aturan trimming berbeda-beda. Tanpa definisi tepat, hasil tidak dapat direproduksi.

**Ganti dengan**
> *Sleep onset* = epoch pertama berlabel N1–REM; *sleep offset* = epoch terakhir berlabel N1–REM. Epoch dipertahankan pada rentang [onset − 60 epoch, offset + 60 epoch] (±30 menit). Epoch Wake di luar rentang dibuang. Efek: SC-78 turun dari 414.961 → 195.469 epoch berlabel (Wake 285.433 → 65.941); ST nyaris tidak berubah (42.691 → 42.467).

---

## 5. Pre-processing #2 — AASM Label Mapping 🟡

**Teks sekarang:** "…menggabungkan Stage 3 dan 4 menjadi N3, serta melakukan drop pada label Movement Time."

**Tambahkan**
- Label **`Sleep stage ?`** (unscored) juga dibuang: 1.335 epoch SC, **2.672 epoch ST (5,9%)**.
- **Crop label ke panjang sinyal**: `n_epoch = floor(n_sampel / 3000)`. Hypnogram lebih panjang dari sinyal pada 150/153 rekaman SC dan lebih pendek pada 38/44 rekaman ST. Tanpa ini indeks epoch bisa bergeser.
- **Aturan memasangkan file:** `SC4001E0-PSG.edf` berpasangan dengan `SC4001EC-Hypnogram.edf`. Karakter ke-8 hypnogram berbeda-beda (kode petugas); cocokkan **7 karakter pertama** (`SC4001E`), bukan nama file lengkap.
- Cocokkan jumlah epoch per kelas dengan tabel referensi sebagai *unit test* (SC-78 trim: N1 = 21.522, N2 = 69.132, N3 = 13.039, REM = 25.835).

---

## 6. Pre-processing #3 — Upsampling EMG 🔴

**Teks sekarang**
> *upsampling* pada EMG submental di SC (dari 1 Hz RMS envelope menjadi 100 Hz) agar dimensi array sejajar … (3.000 data points/epoch)

**Masalah:** Header EDF membuktikan EMG SC memang envelope (`EMG submental, 1 Hz, prefilter "HP:16Hz Rectification LP:0.7Hz"`). Upsampling 30 nilai menjadi 3.000 **tidak menambah informasi sama sekali**, memboroskan 99% komputasi cabang EMG, dan CNN bisa belajar artefak interpolasi. Arsitektur *multi-branch* justru memungkinkan tiap cabang punya panjang input berbeda.

**Ganti dengan**
> EMG submental pada SC berupa *amplitude envelope* 1 Hz (30 sampel/epoch). Sinyal ini **tidak di-upsample**; ditransformasi `log(clip(x, ε))` lalu diberikan ke cabang EMG ringan (Conv1D kecil atau MLP) dengan input 30 sampel. Kanal EEG dan EOG tetap 3.000 sampel (100 Hz).

**Catatan ST:** EMG ST direkam 100 Hz (prefilter `LP:800Hz HP:0.03Hz`) sehingga mengalami *aliasing* dan tidak setara dengan SC. Untuk uji OOD, ubah ke envelope 1 Hz (rectify → low-pass 0,7 Hz → resample 1 Hz) agar meniru SC.

---

## 7. Pre-processing #4 — Bandpass & Z-Score 🔴

**Teks sekarang**
> *Bandpass Filter* (0.5–30 Hz) … *Z-Score Normalization* per *subject*

**Masalah**
1. Bandpass 0,5 Hz pada envelope EMG **menghapus level tonus (komponen DC)** — padahal informasi atonia REM ada di level itu. Filter 30 Hz pada sinyal 1 Hz juga tidak bermakna (Nyquist = 0,5 Hz).
2. EEG SC sudah ber-*high-pass* 0,5 Hz di hardware; filter 0,5 Hz hanya mengulang. EEG ST hanya 0,03 Hz sehingga perlu penyamaan.
3. Z-score "per subject" bila dihitung pada rekaman 24 jam sebelum trimming akan didominasi data siang hari dan artefak.

**Ganti dengan**
> Filter diterapkan **hanya pada EEG dan EOG** (mis. 0,3–35 Hz). EMG tidak difilter. Normalisasi dilakukan **per rekaman (per malam), setelah trimming**, memakai *robust scaling* (median/IQR) dan *clipping* outlier (mis. ±20), sehingga statistik tidak bocor antar-subjek maupun antar-fold.

---

## 8. Feature Engineering Strategy 🟡

| Teks sekarang | Masalah | Ganti dengan |
|---|---|---|
| "EEG Branch: Ekstraksi *spatial features* otak" | "Spatial" artinya antar-elektroda. CNN 1D per kanal mengekstrak fitur **temporal/morfologis/spektral** | "…fitur temporal dan morfologi gelombang (sleep spindles, K-complexes, delta)" |
| "secara drastis memangkas *computational overhead* karena menghindari 2D Spectrogram" | STFT murah di GPU; SeqSleepNet/XSleepNet berbasis spektrogram tetap ringan | "…menyederhanakan *pipeline* (end-to-end tanpa tahap transformasi terpisah) dan mempertahankan resolusi waktu asli" |
| EMG Branch (setara EEG/EOG) | Input EMG hanya 30 sampel | "EMG Branch: cabang ringan atas envelope 1 Hz untuk mendeteksi atonia" |

**Tambahkan:** Cabang EEG sebaiknya dual-skala (kernel kecil ≈ 0,5 s untuk spindle/K-complex dan kernel besar ≈ 4 s untuk gelombang lambat), mengikuti DeepSleepNet.

---

## 9. Proposed Model Architecture (teks + Gambar 2) 🔴

**Masalah pada gambar & teks**
1. Kotak input: "*Input Raw 1D Signals — 3 Channels × 3000 Points*" → salah untuk EMG (30 titik pada SC, lihat #6).
2. Judul kotak "*Spatial_Feature_Extractor*" dan teks "*Spatial Feature Extractor*" → ganti "**Epoch-level Feature Extractor**".
3. Vektor fitur "192-dimensi (3 × 64)": boleh dipertahankan, tetapi bila cabang EMG dibuat lebih kecil (mis. 32-d) total berubah; tulis angka final setelah arsitektur dibekukan.
4. Panjang sekuens "10 epoch" (5 menit) lebih pendek dari yang lazim (TinySleepNet/SeqSleepNet ≈ 20 epoch). Tidak salah, tapi harus diuji.

**Ganti teks dengan**
> *Epoch-level Feature Extractor*: tiga cabang 1D-CNN (EEG Fpz-Cz 3.000 sampel, EOG 3.000 sampel, EMG envelope 30 sampel) menghasilkan vektor fitur per epoch, digabung (*concatenation*) menjadi vektor gabungan. *Temporal Sequence Modeler*: 2-layer Bi-LSTM atas sekuens L epoch berurutan (L = 10 sebagai konfigurasi awal; L ∈ {10, 20, 30} diuji pada ablation), dengan keluaran *many-to-many* 5 kelas per epoch. Total parameter ≈ 0,6–1,2 M.

---

## 10. Evaluation — Loss Function (Focal Loss) 🟡

**Teks sekarang**
> Fungsi ini secara matematis **menekan penalti pada *majority class* (N2)** dan memberikan bobot *loss* yang sangat besar untuk … *minority class* (N1).

**Masalah:** Focal Loss menurunkan bobot **sampel yang mudah dikenali**, bukan **kelas mayoritas**. N2 yang sulit tetap berbobot besar; N1 yang mudah tetap ditekan. Pembobotan per kelas berasal dari faktor **α**, bukan dari γ. Klaim "bobot yang sangat besar untuk N1" juga tidak otomatis benar.

**Ganti dengan**
> Fungsi *loss* diganti dari *Cross-Entropy* standar menjadi **Focal Loss** `FL(p_t) = −α_t (1 − p_t)^γ log(p_t)`. Faktor (1 − p_t)^γ menurunkan kontribusi sampel yang sudah mudah diklasifikasi (umumnya N2/Wake), sehingga gradien lebih terfokus pada sampel sulit (umumnya N1 dan transisi); faktor α_t memberi bobot per kelas. Efektivitasnya dibandingkan secara terkontrol terhadap CE dan *weighted* CE pada protokol validasi yang sama.

**Tambahkan di rencana eksperimen:** CE vs weighted-CE vs Focal(γ = 1, 2) ± α.

---

## 11. Evaluation — Validasi & Metrik 🟡

1. **Typo:** "menjamin 100\\% *epochs*" → tulis `100%` (karakter `\\` ikut terbawa dari export).
2. **Tambahkan definisi data yang dipakai** (lihat #1): "SC-78 (153 rekaman, 78 subjek) = benchmark utama, 10-fold subject-wise; SC-20 (39 rekaman, 20 subjek, usia 25–34) = iterasi cepat; ST = uji generalisasi saja."
3. **GroupKFold harus berdasarkan Subject ID** (bukan nama rekaman): satu subjek = 2 malam (SC4001 & SC4002 = subjek 00). Subjek 13, 36, 52 hanya 1 malam.
4. **Tambahkan validation split di dalam fold training** (berbasis subjek, ±10%) untuk *early stopping* dan pemilihan hiperparameter. Tanpa ini, *checkpoint terbaik* dipilih dari fold uji → hasil optimistik.
5. **Metrik tambahan:** F1 per kelas (terutama N1), confusion matrix gabungan semua fold, rata-rata ± SD antar-fold, ACC.
6. **Judul bagian:** "Evaluation *Matrix*" → "Evaluation *Metrics*" (matrix ≠ metrics).
7. **Kalimat Macro-F1** "bobot kontribusi 20% merata per kelas" sudah benar untuk 5 kelas.
8. **Cohen's κ** "kesepakatan AI vs dokter": tambahkan konteks plafon — kesepakatan antar-scorer manusia ±82,6% overall, N1 ±63% (Rosenberg & Van Hout, 2013), sehingga F1-N1 di atas ±60% tidak realistis.

**Tambahkan baris eksperimen leakage** (menjawab T4): jalankan satu konfigurasi yang sama dengan `KFold(shuffle=True)` *epoch-wise* vs `GroupKFold` subject-wise untuk menunjukkan besarnya inflasi.

---

## 12. End-to-End Pipeline (Gambar 3) 🟡

Gambar sekarang: Raw EDF → Pre-processing (Trimming, Filter, Z-Score) → Subject-Wise GroupKFold → Training (ED-CNN + BiLSTM w/ Focal Loss) → Inference → Evaluation.

**Yang kurang**
1. Tidak ada **validation set** (lihat #11.4).
2. Tidak ada **uji OOD** pada ST dan tidak ada **ablation**.
3. Kotak pre-processing tidak membedakan EEG/EOG (filter) dan EMG (log-envelope) — lihat #6–#7.
4. Pastikan normalisasi bersifat **per-rekaman** (agar aman dari *leakage* bila dilakukan sebelum split).
5. Tidak ada langkah **crop label ke panjang sinyal** dan **pairing PSG–Hypnogram**.
6. Tulisan di kotak Training tampak "ED-CNN" (gambar beresolusi rendah, saya tidak bisa memastikan) — jika memang begitu, ganti menjadi "1D-CNN".

**Alur usulan**
```
EDF (PSG + Hypnogram) → pair by 7-char ID → crop label ke panjang sinyal → buang ?/MT → map S3+S4→N3
  → trim ±30 mnt → [EEG,EOG: filter] [EMG: log-envelope] → normalisasi per-rekaman (robust)
  → GroupKFold(subject) → (train | val subjek untuk early stopping | test)
  → Train: Multi-branch 1D-CNN + 2L Bi-LSTM + Focal Loss
  → Evaluasi gabungan (MF1, κ, F1-N1, CM) → Ablation → Uji OOD (ST)
```

---

## 13. Table Research Gap 🟡

**Teks sekarang:** gap = unimodal, over-engineered, data leakage, N1 rendah; usulan = Multi-Branch 1D-CNN + Bi-LSTM + Focal Loss.

**Masalah**
- Kombinasi bagian-bagiannya **sudah pernah ada**: MMASleepNet (2022) = EEG+EOG+EMG + subject-grouped k-fold; CoSleepNet (2023) = CNN-LSTM + Focal Loss + subject-wise (EEG-EOG). Klaim kebaruan sebaiknya bukan "arsitektur baru".
- "Arsitektur *over-engineered*" adalah penilaian subjektif; perlu bukti (jumlah parameter, waktu) — idealnya berasal dari eksperimen Anda sendiri.

**Ganti dengan**
> Penelitian ini mengisi celah berupa **belum adanya studi terkontrol** yang mengisolasi kontribusi modalitas (EEG, EOG, EMG), fungsi loss (CE, weighted CE, Focal Loss), dan konteks temporal (CNN saja vs CNN + Bi-LSTM) terhadap Macro-F1 dan F1-N1 pada Sleep-EDF dengan validasi *subject-wise* yang seragam, serta menguji generalisasi ke populasi lain (lansia pada SC-78 dan subjek ST dengan temazepam).

**Perbaikan di spreadsheet (Ref-2) yang terkait**
- DOI CoSleepNet di spreadsheet (`10.1109/TNSRE.2021.3076234`) adalah DOI **AttnSleep (Eldele)**. DOI yang benar CoSleepNet: **`10.1016/j.bspc.2022.104299`** (Biomed. Signal Process. Control 80, 104299, 2023).
- Kolom "Metode yang akan digunakan" saling bertentangan (Wavelet db4, VMD+FFT, auxiliary classifier, GELU+residual, LSTM vs Bi-LSTM) dan **tidak konsisten dengan Ref-1** yang menolak Wavelet/VMD. Pilih satu desain final; sisanya masuk *related work / future work*.

---

## 14. Gambar 1 & kerapian 🟢

- **Gambar 1** (screenshot EDFbrowser, resolusi rendah, tanpa keterangan) tidak menyampaikan hasil EDA. Ganti/tambahkan: (a) bar chart distribusi kelas sebelum vs sesudah trimming, (b) contoh hypnogram satu malam, (c) satu epoch EEG/EOG/EMG tiap kelas. Beri keterangan gambar ("Gambar 1. …"). Saya bisa membuat grafik-grafik ini dari data.
- Tanda `**:**` di dalam item (mis. "EEG Branch**:**") merusak format tebal.
- Konsistenkan penulisan: "Sleep-Cassette/SC", "Sleep-Telemetry/ST", "SC-20", "SC-78".
- Kalimat pembuka "7 poin utama": hitung ulang bila bagian ditambah.
- Tambahkan sumber untuk klaim angka literatur (mis. F1-N1 40–56% → Liu et al. 2024).

---

## Bagian yang sudah benar (tidak perlu diubah)

- Pemetaan R&K → AASM (S3+S4 → N3, drop MT).
- Ide trimming Wake ±30 menit; alasan *Wake bias* karena perekaman siang hari.
- GroupKFold berbasis subjek untuk mencegah *data leakage*; pemisahan 2 malam per subjek.
- Pemilihan Macro-F1 dan Cohen's κ sebagai metrik utama.
- Pemisahan cabang per modalitas (*multi-branch*) dan penggunaan EMG untuk membedakan REM vs Wake.
- Bi-LSTM untuk konteks inter-epoch — didukung data: 89,8% epoch memiliki label sama dengan epoch berikutnya.
- Struktur 7 poin dan rujukan ke spreadsheet research gap.
