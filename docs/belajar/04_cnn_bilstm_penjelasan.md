# Penjelasan Lengkap Fase 3 (`04_cnn_model.ipynb`) — Multi-Branch 1D-CNN + Bi-LSTM pada SC-20

Dokumen ini menjelaskan alur kerja, keputusan arsitektur, dan cara membaca hasil pada **`notebooks/04_cnn_model.ipynb`** (Fase 3) sebagai pembanding utama terhadap *baseline* klasik (`03_baseline.ipynb`). Penjelasan disusun berurutan mengikuti **Bagian A sampai Bagian G** di dalam notebook agar mudah dipahami oleh seluruh anggota tim.

---

## 1. Tujuan Fase 3 dan Posisinya dalam Alur Proyek

Pada **Fase 2 (`03_baseline.ipynb`)**, tim telah membangun model klasik (*gradient boosting* di atas fitur spektral & statistik buatan tangan) pada subset **SC-20** (subjek `00` sampai `19`, 39 rekaman malam, $42.307$ *epoch* berlabel valid) dengan tolok ukur:

* **Accuracy:** `84.1%`
* **Macro-F1:** `79.2%`
* **Cohen's Kappa ($\kappa$):** `0.784`
* **F1 kelas N1:** `49.3%`

Pada **Fase 3 (`04_cnn_model.ipynb`)**, kita beralih dari fitur buatan tangan (*hand-crafted features*) ke pendekatan ***end-to-end deep learning*** menggunakan **Multi-Branch 1D-CNN + 2-Layer Bi-LSTM + Focal Loss**.

**Mengapa prototipe dilakukan di SC-20 terlebih dahulu sebelum SC-78?**
1. **Iterasi cepat:** Subset SC-20 berisi 39 rekaman (20 subjek) dibandingkan 153 rekaman (78 subjek) di SC-78, sehingga validasi 20-*fold* dapat diselesaikan dalam hitungan menit pada GPU.
2. **Perbandingan adil (*apple-to-apple*):** Seluruh eksperimen pembanding di literatur Sleep-EDF (seperti DeepSleepNet dan AttnSleep) juga melaporkan skor khusus SC-20 selain SC-78.

---

## 2. Bedah Alur Kerja per Bagian (Bagian A – Bagian G)

### Bagian A — Konfigurasi dan Pengecekan Perangkat
* **Apa yang dilakukan:**
  * Memuat pustaka `torch`, `scikit-learn`, `numpy`, dan `pandas`.
  * Menetapkan *random seed* (`42`) pada Python, NumPy, dan PyTorch agar eksperimen bersifat reproduktif (*reproducible*).
  * Mendeteksi ketersediaan GPU (`torch.cuda.is_available()`).
* **Mengapa penting:**
  * Jika dijalankan di laptop dengan GPU NVIDIA (misalnya RTX 3050), notebook otomatis mengaktifkan akselerasi CUDA dan *Automatic Mixed Precision* (AMP). Jika dijalankan di laptop tanpa GPU (CPU), notebook tetap berjalan aman tanpa *error*.

---

### Bagian B — Memuat Data Terproses SC-20
* **Apa yang dilakukan:**
  * Membaca file manifes `results/tables/02_cache_manifest.csv` hasil praproses Fase 1 (`02_preprocessing.ipynb`).
  * Menyaring baris dengan `subject < 20` (39 berkas `.npz` dari subjek `00` hingga `19`).
  * Memuat sinyal yang sudah ternormalisasi per malam menggunakan median dan *Interquartile Range* (IQR):
    * **EEG (`Fpz-Cz` dan `Pz-Oz`):** dimensi $(N_{\text{epoch}}, 2, 3000)$ pada *sampling rate* $100\text{ Hz}$ ($30\text{ detik} \times 100\text{ Hz} = 3.000$ titik).
    * **EOG (`horizontal`):** dimensi $(N_{\text{epoch}}, 1, 3000)$ pada *sampling rate* $100\text{ Hz}$.
    * **EMG (`submental`):** dimensi $(N_{\text{epoch}}, 1, 30)$ pada *sampling rate* $1\text{ Hz}$ ($30\text{ detik} \times 1\text{ Hz} = 30$ titik).
    * **Label tahap tidur (`labels`):** `0` (Wake), `1` (N1), `2` (N2), `3` (N3), `4` (REM), dan `-1` (`IGNORE_INDEX` untuk *epoch* tak berlabel/artefak gerakan di tengah rekaman).
* **Mengapa `IGNORE_INDEX = -1` dipertahankan?**
  * Menghapus *epoch* berlabel `-1` dari tengah-tengah rekaman akan memutus kesinambungan waktu antar-*epoch* bersebelahan. Dengan menyimpannya sebagai `-1` dan mengabaikannya hanya saat menghitung *loss* dan metrik evaluasi, urutan waktu asli sepanjang malam tetap utuh.

---

### Bagian C — Pembentukan Urutan Sekuensial (`SleepSequenceDataset`)
* **Apa yang dilakukan:**
  * Memotong setiap rekaman malam menjadi jendela berurutan sepanjang `seq_len = 20` *epoch* ($20 \times 30\text{ detik} = 10\text{ menit}$ tidur) dengan pergeseran `stride = 10` *epoch* saat pelatihan.
* **Mengapa menggunakan urutan 20 *epoch* (bukan 1 *epoch* tunggal)?**
  * Dalam aturan klinis AASM, penentuan tahap tidur sangat bergantung pada konteks sebelum dan sesudahnya (misalnya transisi dari *Wake* $\rightarrow$ *N1* $\rightarrow$ *N2*, atau kemunculan *K-complex* dan *sleep spindle* di *epoch* berdekatan). Jendela 20 *epoch* memberi ruang bagi Bi-LSTM untuk membaca dinamika transisi tersebut.

---

### Bagian D — Arsitektur Multi-Branch 1D-CNN + Bi-LSTM dan Focal Loss

#### 1. Mengapa Multi-Branch 1D-CNN (3 Cabang Terpisah)?
Ketiga modalitas polisomnografi (PSG) memiliki karakter fisik dan frekuensi yang sangat berbeda:
* **Cabang EEG (`2 × 3000`, $100\text{ Hz}$):** Memakai lapisan konvolusi 1D bertingkat dengan *kernel* besar pada lapisan awal (`kernel_size = 50`, setara $0,5\text{ detik}$) untuk menangkap gelombang lambat (*delta* $0,5\text{–}4\text{ Hz}$ pada N3), diikuti *kernel* kecil (`kernel_size = 7` dan `5`) untuk pola cepat (*sleep spindle* $11\text{–}16\text{ Hz}$ pada N2 dan *alpha* $8\text{–}12\text{ Hz}$ pada Wake), lalu diringkas menjadi vektor fitur $128$-dimensi.
* **Cabang EOG (`1 × 3000`, $100\text{ Hz}$):** Dirancang terpisah untuk mendeteksi gerak mata lambat (*slow rolling eye movements* pada N1) dan gerak mata cepat (*rapid eye movements* pada REM), menghasilkan vektor fitur $64$-dimensi.
* **Cabang EMG (`1 × 30`, $1\text{ Hz}$):** Karena hanya memiliki $30$ titik per *epoch*, cabang ini menggunakan *kernel* kecil (`kernel_size = 5` dan `3`) agar tidak kehilangan resolusi waktu dalam mendeteksi penurunan tonus otot (*muscle atonia* pada REM), menghasilkan vektor fitur $32$-dimensi.

#### 2. Feature Fusion + 2-Layer Bi-LSTM
* Ketiga vektor fitur dari setiap *epoch* digabungkan (*concatenation*) menjadi vektor berdimensi $128 + 64 + 32 = 224$.
* Vektor sepanjang urutan $20$ *epoch* kemudian diproses oleh **Bidirectional LSTM 2 lapis** (`hidden_size = 128` per arah, sehingga menghasilkan $256$ fitur kontekstual per *epoch*) yang membaca urutan tidur dari masa lalu ke masa depan (*forward*) maupun sebaliknya (*backward*).

#### 3. Penanganan Ketimpangan Kelas dengan Weighted Focal Loss
Pada data Sleep-EDF, distribusi kelas sangat tidak seimbang: tahap **N2** dan **Wake** mendominasi dataset, sedangkan tahap transisi **N1** hanya sekitar $6\%$ dari total *epoch* (sehingga menjadi kelas paling sulit dikenali). Untuk mengatasinya, digunakan kombinasi dua teknik:
1. **Bobot Kelas (*Inverse Square-Root Class Weights*):**
   $$w_c = \left(\frac{N_{\text{total}}}{N_c}\right)^{0.5}$$
   Bobot dinormalisasi agar rata-ratanya bernilai $1$, sehingga kelas minoritas N1 mendapat penalti kesalahan lebih besar tanpa membuat pelatihan tidak stabil.
2. **Focal Loss ($\gamma = 2.0$):**
   $$\mathcal{L}_{\text{focal}}(p_t) = -w_c (1 - p_t)^{\gamma} \log(p_t)$$
   Faktor $(1 - p_t)^2$ secara otomatis menekan kontribusi *loss* dari sampel yang sudah mudah ditebak dengan yakin (seperti *Wake* panjang atau *N2* jelas) dan memfokuskan pembaruan bobot pada sampel sulit di batas transisi (terutama **N1**).

---

### Bagian E — Fungsi Evaluasi Rekaman Penuh (`predict_records`)
* **Cara kerja inferensi:**
  * Saat validasi dan pengujian, seluruh rekaman malam dipotong menjadi blok-blok sekuensial **tanpa tumpang-tindih** (`stride = seq_len = 20`) ditambah satu jendela akhir untuk sisa *epoch* di ujung malam.
  * Setiap *epoch* berlabel valid ($y \neq -1$) mendapatkan tepat satu prediksi bersih tanpa distorsi rata-rata tumpang-tindih.
* **Metrik yang dihitung:**
  * **Accuracy (`ACC`):** Persentase total prediksi benar di seluruh *epoch* valid.
  * **Macro-F1 (`MF1`):** Rata-rata skor F1 dari kelima kelas (`W`, `N1`, `N2`, `N3`, `REM`) dengan bobot setara — menjadi **metrik utama** karena tidak bias terhadap kelas mayoritas.
  * **Cohen's Kappa ($\kappa$):** Mengukur kesepakatan model dengan anotasi pakar manusia di atas peluang tebakan acak.
  * **F1 per Kelas:** Khususnya memantau **F1-N1** sebagai indikator sensitivitas model terhadap tahap tidur transisi.

---

### Bagian F — Validasi Bebas Kebocoran Data (*Subject-Wise LOSO*)
* **Aturan wajib proyek:** Pemisahan data **harus berdasarkan ID subjek**, tidak boleh mengacak *epoch* secara acak (*random split*), karena dua malam atau dua *epoch* dari orang yang sama memiliki kemiripan pola gelombang otak yang membuat skor validasi tampak tinggi secara semu (*data leakage*).
* **Skema 20-Fold Leave-One-Subject-Out (LOSO):**
  * Pada setiap *fold* $k \in \{0, \dots, 19\}$:
    * **1 subjek uji (`test_subj`):** Benar-benar diisolasi dan baru disentuh sekali setelah pelatihan selesai.
    * **1 subjek validasi (`val_subj`):** Digunakan untuk memantau *validation Macro-F1* tiap *epoch* guna menentukan *early stopping* dan menyimpan bobot terbaik (`best_state`).
    * **18 subjek latih (`train_recs`):** Digunakan untuk melatih bobot jaringan.
  * **Mode Adaptif Perangkat:**
    * Pada mesin dengan **GPU NVIDIA (`USE_CUDA = True`)**, notebook otomatis menjalankan penuh **20 *fold*** dengan `batch_size = 64` dan `max_epochs = 25`.
    * Pada mesin **CPU saja (`USE_CUDA = False`)**, notebook otomatis menjalankan **2 *fold* awal** sebagai *sanity check* ringan agar tidak membebani laptop anggota tim yang tidak memiliki GPU.

---

### Bagian G — Tabel Perbandingan dan Visualisasi
Setelah seluruh *fold* selesai, notebook menghasilkan artefak berikut:
1. **`results/tables/04_sc20_per_subject.csv`:** Rincian `ACC`, `Macro-F1`, `Kappa`, dan `F1` kelima kelas untuk masing-masing subjek (`00`–`19`) sehingga tim dapat menganalisis subjek mana yang paling sulit diprediksi.
2. **`results/tables/04_sc20_comparison.csv`:** Tabel perbandingan langsung antara **Baseline Klasik (Fase 2)** dan **Multi-Branch 1D-CNN + Bi-LSTM (Fase 3)** beserta selisih peningkatannya ($\Delta$).
3. **`results/figures/04_sc20_confusion_and_curve.png`:**
   * **Panel Kiri (*Normalized Confusion Matrix*):** Menampilkan persentase prediksi benar pada diagonal utama dan pola salah klasifikasi antar-tahap (misalnya apakah N1 lebih sering tertukar dengan Wake, N2, atau REM).
   * **Panel Kanan (*Validation Macro-F1 Trajectory*):** Menampilkan kurva perkembangan *Macro-F1* validasi per *epoch* pada *fold* pertama untuk memastikan proses konvergensi berjalan stabil tanpa *overfitting* dini.

---

## 3. Panduan Kolaborasi Tim (GPU vs CPU) & Aturan Git

Agar kolaborasi tim berjalan lancar di atas *branch* `feat/cnn-model`:

* **Bagi anggota yang menjalankan pelatihan penuh di GPU NVIDIA:**
  1. Simpan notebook (`Ctrl + S`) setelah *Run All* selesai dan semua sel Markdown (`Bagian A`–`G`) ter-render rapi.
  2. **Jangan men-commit berkas bobot biner `.pt`** di `results/checkpoints/` ke GitHub karena ukurannya besar; bagikan folder `checkpoints` melalui Google Drive tim apabila rekan lain ingin menguji inferensi secara lokal.
  3. Hanya *commit* berkas kode, dokumentasi, tabel CSV, dan gambar PNG:
     ```bash
     git add notebooks/04_cnn_model.ipynb docs/belajar/04_cnn_bilstm_penjelasan.md results/tables/04_sc20_*.csv results/figures/04_sc20_confusion_and_curve.png
     git commit -m "feat(model): add multi-branch 1D-CNN + Bi-LSTM training pipeline on SC-20"
     git push -u origin feat/cnn-model
     ```
* **Bagi anggota tim dengan laptop CPU saja:**
  * Tidak perlu melatih ulang 20 *fold* dari awal. Cukup lakukan `git pull` untuk melihat hasil tabel di `results/tables/04_sc20_comparison.csv`, grafik di `results/figures/04_sc20_confusion_and_curve.png`, dan output ter-render pada `notebooks/04_cnn_model.ipynb`.