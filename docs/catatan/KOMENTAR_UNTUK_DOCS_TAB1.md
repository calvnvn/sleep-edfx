# Daftar Comment untuk Google Docs (Tab 1) — siap tempel

Dokumen: *KLASIFIKASI TAHAP TIDUR OTOMATIS BERBASIS DEEP LEARNING*, tab 1 (bagian "Automated Sleep Stage Classification using Multimodal Hybrid Deep Learning").

## Cara pakai

1. Di Google Docs, blok teks pada kolom **Teks yang diblok** (cukup potongan awalnya, lalu geser sampai akhir frasa).
2. Tekan **Ctrl+Alt+M** (atau klik ikon comment), lalu tempel isi dari kolom **Isi comment**.
3. Urutan di bawah mengikuti urutan teks di dokumen, dari atas ke bawah.
4. Bagian **Catatan untuk Anda** hanya untuk Anda: penjelasan bahasa mudah supaya Anda siap bila ditanya. Jangan ditempel.

Semua angka berasal dari pembacaan langsung file EDF dan hypnogram di `D:\sleep-edfx` (skrip: `scripts/edfx_eda.py`). Singkatan: **SC-78** = seluruh 78 subjek Sleep-Cassette (153 rekaman); **SC-20** = 20 subjek pertama (39 rekaman).

Prioritas: **Tinggi** = koreksi yang mengubah desain atau angka utama. **Sedang** = penajaman. **Rendah** = istilah atau pelengkap.

---

## 1. Pembuka update — Prioritas: Sedang

**Teks yang diblok:** `Berikut adalah rincian dari 7 poin utama dalam kerangka penelitian ini.`

**Isi comment:**
> Update dari progress sebelumnya: dataset Sleep-EDFx sudah diverifikasi langsung dari 197 file EDF, bukan hanya dari paper. Jumlah epoch per kelas setelah trimming cocok dengan angka literatur (SC-78: N1 21.522, N2 69.132, N3 13.039, REM 25.835), sehingga pembacaan label tervalidasi. Dari verifikasi ini ada beberapa koreksi angka dan satu koreksi desain preprocessing (kanal EMG). Koreksi dicatat sebagai comment pada bagian terkait. Hasil model belum ada; yang selesai baru EDA dan penyempurnaan desain.

**Catatan untuk Anda:** ini ringkasan satu paragraf supaya dosen langsung tahu apa yang berubah. Kalimat terakhir sengaja jujur bahwa belum ada hasil training.

---

## 2. Profil dataset — Prioritas: Sedang

**Teks yang diblok:** `197 PSG records dari 100 unique subjects`

**Isi comment:**
> Rencana penggunaan data: SC-78 (153 rekaman, 78 subjek) sebagai benchmark utama dengan 10-fold subject-wise, SC-20 (39 rekaman, 20 subjek) untuk iterasi cepat, dan ST (44 rekaman, 22 subjek) hanya sebagai uji generalisasi di luar distribusi pelatihan, tidak digabung ke training. Alasannya: hampir semua paper pembanding hanya memakai SC, dan kanal EMG berbeda antara SC dan ST (lihat comment EMG). Catatan: 20 subjek SC-20 semuanya berusia 25–34 tahun, sedangkan SC-78 berusia 25–101 tahun, sehingga skor SC-20 dan SC-78 tidak setara. Mohon arahan apakah pembagian ini sesuai.

**Catatan untuk Anda:** "SC-20 vs SC-78" adalah dua "papan skor" standar di literatur. SC-20 populasinya muda semua, jadi lebih mudah. Menggabung SC dan ST membuat model belajar dari dua jenis alat rekam yang berbeda sekaligus.

---

## 3. Deskripsi ST — Prioritas: Rendah

**Teks yang diblok:** `22 pasien insomnia pada Sleep-Telemetry/ST`

**Isi comment:**
> Koreksi deskripsi: menurut dokumentasi PhysioNet, ST berasal dari studi pengaruh obat temazepam pada 22 subjek sehat dengan kesulitan tidur ringan (mild difficulty falling asleep), direkam satu malam plasebo dan satu malam obat. Jadi bukan pasien insomnia terdiagnosis. Efek obat ini juga alasan ST tidak dicampur dengan SC saat pelatihan.

**Catatan untuk Anda:** istilah "pasien insomnia" bisa dipertanyakan dosen yang paham klinis. Wording yang aman: "subjek dengan kesulitan tidur ringan dalam studi temazepam".

---

## 4. Jumlah epoch — Prioritas: Tinggi

**Teks yang diblok:** `483.428 epochs`

**Isi comment:**
> Koreksi angka: total durasi 3.849,0 jam setara sekitar 461.900 epoch 30 detik (3.849 × 120; dihitung per rekaman: 461.870 epoch penuh), bukan 483.428. Angka 483 ribuan mendekati panjang anotasi hypnogram (±484 ribu), yang pada 150 dari 153 rekaman SC lebih panjang daripada sinyal PSG-nya (contoh SC4001E0: anotasi lebih panjang sekitar 1,9 jam). Pada sekitar 37 rekaman ST yang terjadi sebaliknya (anotasi lebih pendek dari sinyal). Konsekuensi untuk pipeline: label harus dipotong ke panjang sinyal (floor(jumlah sampel / 3000)) dan epoch tanpa label dibuang. Dari 461.870 epoch itu, 457.652 berlabel 5 kelas; sisanya berlabel ? (unscored) atau Movement time.

**Catatan untuk Anda:** dua angka di dokumen lama saling bertentangan (3.849 jam tidak mungkin = 483 ribu epoch). Yang benar dihitung dari sinyal. Kalau ditanya "dari mana angka barunya": jam × 120 epoch per jam, dan hasil hitung dari file.

---

## 5. Proporsi N1 — Prioritas: Tinggi

**Teks yang diblok:** `N1 menjadi minority class dengan proporsi < 5%`

**Isi comment:**
> Koreksi: proporsi ini hanya berlaku pada data penuh sebelum trimming (SC-78: N1 = 5,2%). Setelah wake trimming ±30 menit, yang merupakan pipeline yang kita pakai: SC-78 N1 = 11,0%, SC-20 N1 = 6,6%, ST N1 = 8,6%. Pada SC-78 setelah trimming kelas paling jarang justru N3 (6,7%). Rasio kelas terbesar : terkecil sekitar 5–6 : 1, sehingga imbalance bersifat moderat, bukan ekstrem. N1 tetap kelas tersulit, tetapi karena ambigu (kesepakatan antar-dokter untuk N1 hanya 63,0%, Rosenberg & Van Hout, 2013), bukan hanya karena jarang.

**Catatan untuk Anda:** setelah jam-jam siang dibuang, N1 jadi tampak lebih banyak karena pembaginya mengecil. Pesan utamanya: Focal Loss tetap relevan, tapi alasannya jangan "N1 < 5%".

---

## 6. Wake bias — Prioritas: Tinggi

**Teks yang diblok:** `Wake bias yang sangat masif`

**Isi comment:**
> Rincian angka: Wake = 68,8% pada SC-78 penuh (68,0% pada SC-20), turun menjadi 33,7% (SC-78) dan 19,6% (SC-20) setelah trimming. Pada ST, Wake hanya 10,4% karena direkam khusus jam tidur. Setelah trimming, kelas mayoritas adalah N2 (SC-78 35,4%; SC-20 42,1%; ST 46,7%), bukan Wake. Tabel lengkap sebelum dan sesudah trimming:
> SC-78 sebelum → sesudah: W 68,8 → 33,7 | N1 5,2 → 11,0 | N2 16,7 → 35,4 | N3 3,1 → 6,7 | REM 6,2 → 13,2.
> SC-20 sebelum → sesudah: W 68,0 → 19,6 | N1 2,6 → 6,6 | N2 16,7 → 42,1 | N3 5,4 → 13,5 | REM 7,3 → 18,2.

**Catatan untuk Anda:** komentar ini melengkapi komentar nomor 5. Kalau dosen hanya sempat membaca satu, tabel ini yang paling berguna.

---

## 7. Wake trimming — Prioritas: Sedang

**Teks yang diblok:** `30-Minute Wake Trimming` (kemunculan pertama, di bagian "Pre-processing Pipeline"; frasa yang sama muncul lagi lebih bawah)

**Isi comment:**
> Aturan trimming perlu ditulis eksplisit karena memengaruhi jumlah Wake: kami mempertahankan 30 menit (60 epoch) sebelum epoch tidur pertama dan sesudah epoch tidur terakhir. Dengan aturan ini jumlah Wake SC-78 = 65.941 epoch; paper lain melaporkan 69.824 karena aturan trimming berbeda, sedangkan jumlah N1/N2/N3/REM identik. Trimming dilakukan sebelum normalisasi (lihat comment normalisasi).

**Catatan untuk Anda:** perbedaan Wake antar-paper itu normal. Yang penting aturan Anda tertulis dan konsisten.

---

## 8. Pemetaan label — Prioritas: Rendah

**Teks yang diblok:** `AASM Label Mapping`

**Isi comment:**
> Tambahan: selain Movement time (128 epoch di SC, 83 di ST), label "Sleep stage ?" (unscored) juga dibuang: 1.335 epoch di SC dan 2.672 epoch di ST (5,9% dari seluruh epoch ST). Tahap 3 dan 4 digabung menjadi N3.

**Catatan untuk Anda:** pelengkap kecil. Tanda tanya ("?") bukan stage tidur, jadi tidak boleh ikut dilatih.

---

## 9. Upsampling EMG — Prioritas: Tinggi

**Teks yang diblok:** `Melakukan upsampling pada channel EMG submental di dataset SC`

**Isi comment:**
> Revisi desain. Metadata header EDF (SC4001E0-PSG.edf) menunjukkan EMG submental pada SC disampel 1 Hz dengan prefilter "HP:16Hz Rectification LP:0.7Hz", yaitu sudah berupa amplitude envelope (kekuatan otot per detik), bukan sinyal mentah. Meng-upsample 1 Hz → 100 Hz hanya mengulang/menginterpolasi 30 nilai menjadi 3.000 nilai tanpa menambah informasi, dan memboroskan komputasi cabang CNN. Usulan: cabang EMG menerima 30 sampel per epoch, dinormalisasi robust per rekaman (median/IQR, tanpa log), dengan cabang kecil, sedangkan EEG dan EOG tetap 3.000 titik. Normalisasi per rekaman wajib karena skala envelope berbeda jauh antar-rekaman (median per rekaman berkisar −0,98 sampai 17,5; 55 dari 153 rekaman punya lebih dari 1% nilai negatif), sehingga log-transform tidak aman. Setelah normalisasi, EMG masih informatif: rata-rata epoch tertinggi pada Wake dan terendah pada REM, dan EMG saja memisahkan REM dari Wake dengan AUC 0,84 (pooled, statistik sederhana). Struktur multi-branch tetap valid karena tiap cabang memang boleh punya panjang input berbeda; dimensi fusi (192-d) menyesuaikan.
> Catatan tambahan: pada ST, EMG direkam 100 Hz dengan prefilter LP:800Hz HP:0.03Hz (bukan envelope), sehingga tidak setara dengan EMG SC. Bila ST dipakai sebagai uji generalisasi, EMG ST perlu diubah ke envelope 1 Hz agar meniru SC.

**Catatan untuk Anda:** analogi: EMG mentah itu rekaman suara lengkap, yang tersimpan di SC cuma "angka volume per detik" (alat kaset 1980-an terbatas kapasitasnya). Memperbanyak 30 angka jadi 3.000 angka tidak membuat informasinya bertambah. "Tahu dari mana": dari header file EDF yang dibaca langsung, bukan tebakan. Dokumen Ref-3 sendiri sudah menyebut ini envelope, jadi ini lebih ke konsistensi daripada "kesalahan besar".

---

## 10. Bandpass filter — Prioritas: Tinggi

**Teks yang diblok:** `Bandpass Filter` (kemunculan pertama, di poin "Denoising & Scaling"; frasa yang sama muncul lagi lebih bawah)

**Isi comment:**
> Revisi: bandpass 0,5–30 Hz diterapkan hanya pada EEG dan EOG. Untuk EMG envelope filter ini tidak tepat: informasi pada envelope adalah level (tinggi = otot tegang pada Wake, rendah = atonia pada REM), yaitu komponen sangat lambat yang akan terbuang oleh high-pass 0,5 Hz. Data 1 Hz juga hanya memuat frekuensi sampai 0,5 Hz (Nyquist), sehingga batas 30 Hz tidak bermakna. EMG cukup dinormalisasi robust per rekaman (median/IQR).

**Catatan untuk Anda:** "level" itu seperti volume rata-rata. High-pass membuang "volume rata-rata" dan hanya menyisakan "naik-turunnya", padahal yang membedakan Wake dari REM justru volume rata-ratanya.

---

## 11. Normalisasi — Prioritas: Sedang

**Teks yang diblok:** `Z-Score Normalization per subject`

**Isi comment:**
> Usulan penajaman: normalisasi dihitung per rekaman (per malam), setelah trimming, dengan robust scaling (median/IQR) dan clipping nilai ekstrem. Alasan: (1) jika dihitung pada rekaman penuh SC (±22,7 jam, 68% Wake siang hari), statistiknya didominasi aktivitas siang dan artefak; (2) 75 dari 78 subjek SC punya dua malam dengan pemasangan elektroda terpisah, sehingga amplitudo bisa berbeda antar-malam. Mohon masukan apakah per-rekaman dapat diterima.

**Catatan untuk Anda:** ini usulan, bukan koreksi pasti. Karena itu kalimatnya meminta masukan.

---

## 12. Istilah fitur spasial — Prioritas: Rendah

**Teks yang diblok:** `Ekstraksi spatial features otak`

**Isi comment:**
> Penajaman istilah: 1D-CNN yang bekerja per kanal mengekstraksi fitur temporal/morfologis (bentuk gelombang dan komponen frekuensi), bukan fitur spasial. Istilah spasial lazimnya mengacu pada hubungan antar-elektroda. Usulan: ganti "spatial features" dengan "temporal/morphological features", demikian juga "Spatial Feature Extractor" pada bagian arsitektur.

**Catatan untuk Anda:** salah istilah kecil, tapi mudah ditangkap dosen yang paham sinyal.

---

## 13. Bi-LSTM dan panjang sekuens — Prioritas: Sedang

**Teks yang diblok:** `Temporal Sequence Modeler`

**Isi comment:**
> Justifikasi dari data: pada SC-78 (setelah trimming) 89,8% epoch memiliki label sama dengan epoch berikutnya (stage bertahan: W 94,6%, N2 90,8%, REM 94,2%, N3 79,7%, N1 hanya 72,4%), sehingga konteks antar-epoch sangat informatif. Transisi tidak wajar seperti W→N3 atau N3→REM praktis tidak pernah terjadi. Panjang sekuens 10 epoch (5 menit) relatif pendek dibanding literatur (TinySleepNet memakai 20 epoch); rencananya diuji sebagai ablation (L = 10, 20, 30).

**Catatan untuk Anda:** angka "89,8%" artinya hampir 9 dari 10 epoch sama dengan epoch sesudahnya. Itulah alasan kuantitatif memakai model sekuens.

---

## 14. Focal Loss — Prioritas: Tinggi

**Teks yang diblok:** `Mengganti Cross-Entropy standar dengan Focal Loss`

**Isi comment:**
> Penajaman penjelasan: Focal Loss mengecilkan bobot sampel yang mudah ditebak (suku (1 − p_t)^γ), bukan menekan kelas mayoritas secara langsung. Penyeimbangan antar-kelas berasal dari parameter α per kelas. Karena itu klaim "menekan penalti N2" perlu diganti menjadi "menekan bobot sampel mudah". Perlu dibuktikan secara empiris, bukan diasumsikan: rencana ablation membandingkan CE, weighted CE, dan Focal (γ = 1, 2 dengan/tanpa α) pada protokol validasi yang sama. Catatan literatur: CoSleepNet (Efe & Özşen, 2023) sudah memakai Focal Loss dengan validasi subject-wise, jadi nilai tambahnya ada pada perbandingan terkontrol, bukan pada penggunaan Focal Loss itu sendiri.

**Catatan untuk Anda:** bayangkan Focal Loss "tidak peduli pada soal yang sudah gampang dijawab". Soal N1 biasanya sulit, jadi otomatis dapat perhatian lebih. Membuat kelas langka dapat bobot lebih adalah tugas α.

---

## 15. Validasi subject-wise — Prioritas: Sedang

**Teks yang diblok:** `Subject-Wise GroupKFold Cross-Validation (berbasis Subject ID)`

**Isi comment:**
> Pilihan ini tepat; detail pelaksanaan yang akan dipegang: (1) kelompok = ID subjek, bukan ID rekaman, karena 75 dari 78 subjek SC punya dua malam (SC13, SC36, SC52 hanya satu); (2) di setiap fold, sebagian subjek training disisihkan sebagai validasi untuk early stopping dan pemilihan hyperparameter, agar fold uji tidak dipakai untuk tuning; (3) hasil dilaporkan dari prediksi gabungan semua fold dan sebagai rata-rata ± simpangan baku per fold; (4) satu eksperimen sengaja epoch-wise untuk mengukur besar kebocoran data (leakage). Catatan format: teks "100\% epochs" pada paragraf ini tampil dengan tanda \ yang tidak diperlukan.

**Catatan untuk Anda:** poin (2) sering terlewat dan sering dipertanyakan reviewer. Poin (4) dapat menjadi hasil menarik tersendiri.

---

## 16. Metrik dan target — Prioritas: Sedang

**Teks yang diblok:** `Primary Metrics`

**Isi comment:**
> Tambahan pelaporan: selain Macro F1 dan Cohen's κ, dilaporkan F1 per kelas (terutama N1), akurasi, dan confusion matrix gabungan. Target realistis sebagai acuan: SC-20 Macro F1 78–82 dan SC-78 75–79, F1-N1 sekitar 45–52. Pembanding: TinySleepNet MF1 80,5 / κ 0,80 / F1-N1 51,4 (SC-20) dan 78,1 / 0,77 / 51,0 (SC-78); MMASleepNet (EEG+EOG+EMG) MF1 82,65 (SC-20) dan 77,60 (SC-78). Batas praktis ditentukan kesepakatan antar-dokter: rata-rata 82,6%, N1 63,0%, N3 67,4% (Rosenberg & Van Hout, 2013), sehingga F1-N1 di atas 60% tidak realistis diharapkan.

**Catatan untuk Anda:** angka pembanding ini dikutip dari paper masing-masing sebagaimana dikompilasi di WaveSleepNet (2024) dan Ref-1 Anda. Jangan menyebutnya hasil Anda sendiri; protokol tiap paper bisa sedikit berbeda.

---

## 17. Research gap — Prioritas: Tinggi

**Teks yang diblok:** `Rincian perbandingan antara pendekatan model-model sebelumnya`

**Isi comment:**
> Penajaman posisi penelitian: CoSleepNet (2023) sudah memakai CNN-LSTM + Focal Loss + validasi subject-wise (EEG-EOG), dan MMASleepNet (2022) sudah memakai EEG+EOG+EMG dengan fold yang dikelompokkan per subjek. Sehingga gap yang tepat bukan "belum ada yang menggabungkan", melainkan: belum ada studi yang secara terkontrol (protokol seragam) memisahkan kontribusi modalitas (EEG, EOG, EMG), fungsi loss (CE, weighted CE, Focal), dan konteks temporal terhadap F1-N1 dan Macro F1 pada Sleep-EDF, disertai uji generalisasi ke populasi berbeda (lansia pada SC-78 dan ST). Dengan begitu kontribusi diposisikan sebagai controlled ablation and generalization study.
> Perbaikan pada spreadsheet: DOI CoSleepNet yang benar adalah 10.1016/j.bspc.2022.104299 (DOI 10.1109/TNSRE.2021.3076234 milik AttnSleep). Kolom "metode yang akan digunakan" juga perlu diseragamkan (Wavelet/VMD pada beberapa baris bertentangan dengan keputusan memakai sinyal 1D mentah; LSTM vs Bi-LSTM).

**Catatan untuk Anda:** ini komentar terpenting untuk sidang. Kalau gap dinyatakan terlalu kuat, satu penguji yang tahu CoSleepNet bisa menjatuhkannya. Rumusan "ablation terkontrol" jauh lebih aman.

---

## 18. Pertanyaan untuk konsultasi — Prioritas: Tinggi

**Teks yang diblok:** `Kelompok8_ResearchGap_Assignment`

**Isi comment:**
> Hal yang ingin dikonsultasikan:
> 1. Apakah kontribusi boleh diposisikan sebagai controlled ablation and generalization study (modalitas × loss × konteks temporal × skema validasi)?
> 2. Apakah pembagian data (SC-78 utama, SC-20 iterasi cepat, ST hanya uji generalisasi) sudah sesuai?
> 3. Baseline pembanding apa yang diharapkan? Usulan: TinySleepNet (re-implementasi) dan baseline fitur + gradient boosting, dengan GPU RTX 3050 6 GB.
> 4. Apakah target Macro F1 sekitar 75–79 (SC-78) dan F1-N1 sekitar 45–52 dianggap memadai?
> 5. Bentuk luaran akhir (laporan, paper, atau demo) dan tenggatnya, untuk menentukan seberapa luas ablation yang realistis.
> Langkah berikutnya: setup environment (PyTorch, MNE), preprocessing ke satu file per rekaman, baseline, lalu ablation di SC-20 sebelum SC-78 dan uji ST.

**Catatan untuk Anda:** komentar ini sekaligus agenda konsultasi. Anda bisa langsung membacakannya.

---

## Urutan tempel yang disarankan bila waktu terbatas

Bila hanya sempat menempel sebagian, pilih yang berprioritas **Tinggi**: nomor 4, 5, 6, 9, 10, 14, 17, 18. Delapan comment itu sudah mencakup semua koreksi inti dan seluruh pertanyaan konsultasi.

## Yang belum ada (jangan dijanjikan ke dosen)

- Belum ada hasil model apa pun (baik deep learning maupun baseline). Eksperimen baseline gradient boosting yang sempat dijalankan belum selesai, jadi tidak ada angkanya di sini.
- Angka pembanding paper (TinySleepNet, MMASleepNet, dll.) dikutip dari sumber sekunder atau Ref-1, belum diperiksa ulang satu per satu ke paper aslinya.
