# Notebook 04: CNN + Bi-LSTM pada 20 Subjek Muda, Bahan Presentasi

Dokumen ini untuk tim yang akan mempresentasikan `notebooks/04_cnn_bilstm_sc20.ipynb`. Istilah mengikuti tabel istilah di notebook 01.

---

## 1. Pesan utama

1. **Model.** Model *deep learning* menerima 20 *epoch* berturut-turut (10 menit). Tiga cabang CNN meringkas EEG, EOG, dan EMG setiap *epoch*, lalu Bi-LSTM membaca urutannya maju dan mundur.
2. **Hasil pada 20 subjek muda.** Macro-F1 mencapai 80,1 ± 0,4 dengan *cross-entropy* dan 80,2 ± 0,9 dengan *focal loss*, dibandingkan 79,2 untuk *baseline*. F1-N1 naik sekitar 4 poin.
3. **Belum terbukti per subjek.** Uji per subjek tidak bermakna (p ≥ 0,20). Dengan 20 subjek, keunggulan sekitar 1 poin belum bisa dibuktikan. Karena itu model diuji ulang pada 78 subjek di notebook 05, tanpa mengubah pengaturan.
4. ***Cross-entropy* vs *focal loss*.** Keduanya setara pada Macro-F1, tetapi profil kesalahannya berbeda. *Focal loss* di sini juga memakai bobot kelas, jadi dua faktor berubah sekaligus. Keduanya dipisahkan di notebook 06.

## 2. Posisi dalam proyek

Notebook 03 membuktikan bahwa setiap sinyal dan konteks tetangga menambah skor. Notebook 04 menguji apakah model yang mempelajari ciri sendiri dari sinyal mentah, dan membaca urutan dengan Bi-LSTM, bisa lebih baik lagi. Pembagian data dan subjek validasinya sama persis dengan *baseline*.

Notebook ini adalah **versi perombakan**. Versi pertama mencapai Macro-F1 78,6, di bawah *baseline*. Peninjauan menemukan kelemahan yang membuat hasilnya sulit ditafsirkan:

| Masalah pada versi pertama | Perbaikan |
|---|---|
| Rumus *focal loss* salah: bobot kelas ikut menjadi pangkat | Rumus dibetulkan dan diuji dengan tiga cara |
| Tidak ada pembanding *cross-entropy* | *Cross-entropy* dijalankan dengan pengaturan yang sama |
| Validasi hanya 1 subjek | 2 subjek, sama dengan *baseline* |
| Dua jalur kernel EEG tidak benar-benar paralel | Dua jalur paralel dalam satu cabang |
| Hanya dijalankan sekali | 3 *seed* dan uji per subjek |

Penyebab pasti hasil buruk versi pertama tidak bisa dipisahkan, karena beberapa hal diubah sekaligus.

## 3. Alur presentasi

| Urutan | Yang ditampilkan | Yang diucapkan | Angka kunci |
|---|---|---|---|
| 1 | Diagram arsitektur (bagian D) | Tiga cabang CNN untuk tiga jenis sinyal, lalu Bi-LSTM untuk konteks | 1,09 juta parameter, 70% di Bi-LSTM |
| 2 | Tabel hasil (bagian G) | Sekitar 1 poin di atas *baseline*; variasi antar-*seed* hampir sebesar itu | 80,1 ± 0,4 vs 79,2 |
| 3 | Grafik per subjek (H1) | Keunggulan belum terbukti per subjek, dan hasilnya berbeda-beda antar-orang | p = 0,475; 12 dari 20 subjek |
| 4 | Matriks kebingungan (H2) | *Cross-entropy* kuat di kelas besar; *focal loss* dengan bobot kelas memulihkan N1 dan N3 | N1 50,5 vs 60,1 |

Bagian E (uji rumus *focal loss*), F (subjek validasi identik), H3 (letak kesalahan Wake), dan H4 (kurva latih) disiapkan untuk menjawab pertanyaan.

## 4. Penjelasan per bagian

### A–B. Pengaturan dan data

**Pengaturan.**

| Aspek | Nilai | Alasan |
|---|---|---|
| Panjang jendela | 20 *epoch* (10 menit) | Cukup konteks dan muat di GPU 6 GB. Panjang lain diuji di notebook 06. |
| Geser jendela | 10 *epoch* | Separuh tumpang tindih: lebih banyak contoh latih, dan *epoch* di tepi jendela mendapat beberapa tebakan |
| Optimasi | AdamW, laju belajar 0,001, laju diturunkan perlahan (kosinus), gradien dibatasi 1,0 | Pengaturan umum yang stabil; tidak disetel |
| Penghentian | Macro-F1 validasi, berhenti setelah 6 putaran tanpa perbaikan, maksimum 30 | Selaras dengan metrik utama |
| Pengulangan | 3 *seed* × 2 *loss* × 20 *fold* | Mengukur variasi acak |

**Data.**
- 39 rekaman, 42.307 *epoch* berlabel. Jumlah ini dicocokkan dengan notebook 02.
- N1 hanya 6,6% dari seluruh *epoch*.

### C. Jendela 20 *epoch*

**Cara kerja.**
- Setiap rekaman dipotong menjadi jendela 20 *epoch* yang digeser 10 *epoch*.
- Ada jendela tambahan yang berakhir tepat di ujung rekaman, supaya akhir malam tidak terlewat.
- Jendela tidak pernah melintasi dua rekaman.
- *Epoch* berlabel −1 tetap ada di jendela, tetapi tidak dinilai.
- Saat pengujian, peluang dari semua jendela yang menutupi satu *epoch* dirata-ratakan.

**Keluaran.** Rekaman contoh (841 *epoch*) menghasilkan 84 jendela.

### D. Arsitektur

```
EEG (2 kanal, 3.000 titik) ─► cabang dua skala ─► 128 ┐
EOG (1 kanal, 3.000 titik) ─► cabang dua skala ─►  64 ├─► gabung 224 ─► Bi-LSTM 2 lapis ─► 5 skor per epoch
EMG (selubung, 30 titik)   ─► konvolusi kecil  ─►  32 ┘
```

**Cabang dua skala (EEG dan EOG).** Ada dua jalur paralel pada sinyal yang sama:
- **Jalur pendek:** konvolusi dengan kernel 50 sampel (0,5 detik), peka pada gelombang cepat seperti alfa dan *spindle*.
- **Jalur panjang:** konvolusi dengan kernel 400 sampel (4 detik), peka pada gelombang lambat.

Setiap jalur diakhiri dengan rerata sepanjang waktu (*global average pooling*), lalu keduanya digabung. Rancangan ini mengikuti DeepSleepNet (Supratak dkk., 2017).

**Komponen lain.**
- **Cabang EMG:** dua konvolusi kecil. Masukannya hanya 30 titik, jadi kernel besar tidak bermakna.
- **Bi-LSTM:** 2 lapis, 128 unit per arah. Bagian inilah yang membaca konteks waktu.

**Parameter.**

| Komponen | Parameter |
|---|---|
| Cabang EEG | 262.400 |
| Cabang EOG | 65.872 |
| Cabang EMG | 1.712 |
| Bi-LSTM | 757.760 |
| Kepala klasifikasi dan normalisasi | 1.733 |
| **Total** | **1.089.477** |

**Makna.** Model ini kecil dibandingkan model di literatur, tetapi sangat besar dibandingkan 17 subjek latih per *fold*. Risiko model menghafal subjek latih diawasi lewat data validasi.

### E. *Loss* dan bobot kelas

**Rumus.** Untuk satu *epoch* dengan tahap benar *y*, *p_t* adalah peluang yang diberikan model pada tahap yang benar:
- ***Cross-entropy*:** −α_y · log(*p_t*).
- ***Focal loss*:** −α_y · (1 − *p_t*)^γ · log(*p_t*), dengan γ = 2 (Lin dkk., 2017). Faktor (1 − *p_t*)^γ mengecilkan hukuman untuk *epoch* yang sudah mudah ditebak.

**Bobot kelas** (α) berbanding terbalik dengan akar frekuensi kelas: Wake 0,89, N1 1,52, N2 0,60, N3 1,07, REM 0,92. Akar dipakai supaya bobotnya moderat.

**Tiga uji rumus.**
1. Dengan γ = 0 dan tanpa bobot, *focal loss* sama persis dengan *cross-entropy*.
2. Hasilnya cocok dengan hitungan tangan.
3. *Epoch* berlabel −1 tidak memengaruhi *loss*.

**Hal penting.** *Cross-entropy* di notebook ini dijalankan tanpa bobot kelas, sedangkan *focal loss* dengan bobot kelas. Jadi perbandingan keduanya mengubah dua hal sekaligus.

### F. Subjek validasi dan uji kewarasan

**Subjek validasi.** Di setiap *fold*, 2 dari 19 subjek non-uji dijadikan data validasi. Pemilihannya diperiksa otomatis dan identik dengan *baseline* di ke-20 *fold*.

**Uji kewarasan.** Model harus mampu menghafal 16 jendela dengan sempurna, dan hasilnya 100%. Kalau ada kesalahan pada data, label, atau gradien, model tidak akan mampu menghafal data sekecil itu.

Bobot yang disimpan adalah bobot pada putaran dengan Macro-F1 validasi terbaik, bukan bobot terakhir.

### G. Pelatihan dan hasil

**Macro-F1 per *seed*.**

| *Loss* | *Seed* 0 | *Seed* 1 | *Seed* 2 |
|---|---|---|---|
| *Cross-entropy* | 80,4 | 80,1 | 79,7 |
| *Focal loss* | 80,9 | 79,2 | 80,3 |

Setiap kombinasi butuh 23–31 menit, sehingga total sekitar 2,8 jam.

**Ringkasan.**

| | *Cross-entropy* | *Focal loss* | *Baseline* |
|---|---|---|---|
| Akurasi | 85,1 ± 0,1 | 84,5 ± 0,6 | 84,1 |
| Macro-F1 | 80,1 ± 0,4 | 80,2 ± 0,9 | 79,2 |
| κ | 0,795 ± 0,001 | 0,790 ± 0,008 | 0,784 |
| F1-N1 | 53,0 ± 1,1 | 53,4 ± 2,4 | 49,3 |
| Macro-F1 validasi | 82,5 ± 0,2 | 82,4 ± 0,1 | – |

**Makna.**
- Model sekitar 1 poin di atas *baseline*, tetapi selisih ini hanya sekitar dua kali simpangan baku antar-*seed*.
- *Cross-entropy* terpilih berdasarkan skor validasi (82,5 vs 82,4). Selisih ini praktis seri, jadi pilihan ini sebaiknya dibaca sebagai memilih yang lebih sederhana dan lebih stabil.
- Skor validasi lebih tinggi daripada skor uji. Dugaan penyebabnya ada dua, dan belum diuji:
  - skor validasi diambil dari putaran terbaik, yang dipilih justru karena tertinggi;
  - data validasi hanya berisi 2 subjek.

### H1. Uji per subjek

| Perbandingan | Rerata selisih | Subjek lebih baik | p Wilcoxon |
|---|---|---|---|
| *Cross-entropy* vs *baseline* | +0,64 | 12 dari 20 | 0,475 |
| *Focal loss* vs *baseline* | +0,56 | 14 dari 20 | 0,202 |
| *Focal loss* vs *cross-entropy* | −0,07 | 11 dari 20 | 0,927 |

**Makna.**
- Tidak ada perbandingan yang bermakna.
- Hasil per subjek bervariasi:
  - subjek 2 naik dari 72,0 menjadi 82,6;
  - subjek 5 turun dari 84,0 menjadi 79,6;
  - subjek 11 dan 17 sulit bagi semua model.
- Perbandingan dilakukan berpasangan pada subjek yang sama. Cara ini menghapus perbedaan tingkat kesulitan antar-orang, yang jauh lebih besar daripada selisih antar-model.

### H2. Jenis kesalahan per tahap

**Sensitivitas (%).**

| | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| *Baseline* | 87,5 | 55,3 | 85,0 | 86,1 | 87,3 |
| *Cross-entropy* | 89,4 | 50,5 | 88,1 | 81,2 | 89,0 |
| *Focal loss* | 87,9 | 60,1 | 84,9 | 85,0 | 88,6 |

**Makna.**
- *Cross-entropy* tanpa bobot lebih baik pada kelas besar. Sebanyak 23% N1 dan 18% N3 terbaca sebagai N2.
- *Focal loss* dengan bobot memulihkan N1 dan N3, dengan harga Wake dan N2 kembali ke tingkat *baseline*.
- Ini tukar-tambah antar-kelas, bukan perbaikan menyeluruh. Pola yang sama muncul lagi pada 78 subjek di notebook 05, dan penyebabnya diuji di notebook 06.

### H3. Letak kesalahan Wake

**Persentase *epoch* Wake yang salah ditebak.**

| | Sebelum tidur | Tengah malam | Setelah tidur |
|---|---|---|---|
| *Baseline* | 4,8 | 23,7 | 2,9 |
| *Cross-entropy* | 3,9 | 18,6 | 4,9 |
| *Focal loss* | 4,6 | 22,5 | 3,5 |

**Makna.**
- Sekitar 83% kesalahan Wake *baseline* terjadi di tengah malam, yaitu saat terbangun singkat.
- *Cross-entropy* mengurangi kesalahan di tengah malam, tetapi menambah kesalahan setelah tidur.
- Dugaannya, konteks dua arah membantu mengenali terbangun singkat. Dugaan ini belum diuji terpisah.

### H4. Kurva pelatihan

**Hasil.**
- Putaran terbaik: median 6, rentang 1–30.
- Hanya satu *fold* yang mencapai batas 30 putaran.

**Makna.** Pelatihan tidak dibatasi oleh jumlah putaran. Keterbatasan utamanya adalah jumlah subjek latih yang sedikit. Skor validasi naik-turun 1–3 poin antar-putaran, karena data validasi hanya berisi 2 subjek.

### H5. Contoh hipnogram

Kesepakatan dengan pakar pada satu rekaman: 86,2% untuk *baseline* dan 86,7% untuk model. Ini hanya ilustrasi visual, bukan bukti.

## 5. Keterbatasan yang perlu diakui

- **Subjek muda saja.** Hanya 20 subjek, semuanya berusia 25–34 tahun.
- ***Baseline* dijalankan sekali.** Variasi antar-*seed*-nya tidak diukur.
- **Pengaturan tidak disetel.** Panjang jendela, bobot kelas, dan hiperparameter lain tidak dioptimasi. Sebagian diuji di notebook 06.
- **Dua faktor tercampur.** Perbandingan *cross-entropy* vs *focal loss* mengubah dua hal sekaligus (γ dan bobot kelas).

## 6. Antisipasi pertanyaan

**"Jadi *deep learning* lebih baik atau tidak?"**
- Pada 20 subjek, keunggulannya belum terbukti: sekitar 1 poin, dengan p ≥ 0,20.
- Pada 78 subjek (notebook 05), model lebih baik pada 48 dari 78 subjek (p = 0,005). Namun ada tiga subjek yang gagal parah, sehingga rerata selisihnya belum pasti di atas nol.

**"Kenapa jendela 20 *epoch*?"**
Kompromi antara konteks dan memori GPU 6 GB. Jendela 10 dan 30 *epoch* diuji di notebook 06.

**"Kenapa tiga *seed*?"**
Untuk mengukur seberapa besar hasil berubah karena keacakan. Selisih Macro-F1 antar-*seed* mencapai 1,7 poin (*focal loss*), lebih besar daripada selisih model dengan *baseline*.

**"Apakah ada kebocoran data?"**
Tidak, karena empat hal:
- subjek latih, validasi, dan uji dipastikan terpisah secara otomatis;
- jendela tidak melintasi rekaman;
- pemilihan *loss* memakai data validasi;
- subjek validasi identik dengan *baseline*.

**"Kenapa 70% parameter di Bi-LSTM?"**
Lapisan LSTM dengan 224 masukan dan 128 unit per arah memang besar. Apakah Bi-LSTM benar-benar diperlukan diuji di notebook 06, dengan membuangnya.

**"Kenapa N1 masih rendah?"**
N1 adalah tahap peralihan yang jarang (6,6% data), dan pakar pun hanya sepakat sekitar 63%. Tanpa bobot kelas, model jarang menebak N1. Dengan bobot kelas, sensitivitas N1 naik menjadi 60%.

## 7. Lampiran: membaca kode

| Komponen kode | Fungsinya |
|---|---|
| `SequenceDataset` | Membentuk jendela 20 *epoch* per rekaman |
| `DualScaleBranch` | Cabang CNN dua jalur untuk EEG dan EOG |
| `SleepNet` | Tiga cabang + Bi-LSTM + kepala klasifikasi |
| `FocalLoss` | *Focal loss* dengan bobot kelas sebagai pengali; mengabaikan label −1 |
| `validation_subjects(fold)` | Memilih 2 subjek validasi, identik dengan notebook 03 |
| `train_fold` | Pelatihan satu *fold* dengan penghentian dini |
| `predict_probs` | Prediksi dengan rata-rata jendela yang tumpang tindih |
| `run_cv(loss, seed)` | 20 *fold* untuk satu kombinasi; hasilnya disimpan dan dipakai ulang kalau pengaturan sama |

**Berkas.**
- Tabel: `results/tables/04_cnn_sc20_*.csv`.
- Gambar: `results/figures/04_cnn_sc20_*.png`.
- Prediksi: `data/processed/cnn_sc20_runs/` dan `cnn_sc20_predictions.npz`.

**Keputusan desain.** K6 di `docs/catatan/KEPUTUSAN_DESAIN.md`.

**Rujukan.**
- Supratak dkk. (2017), DeepSleepNet, *IEEE TNSRE* 25(11).
- Lin dkk. (2017), *Focal Loss for Dense Object Detection*, ICCV.
