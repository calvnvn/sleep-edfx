# Notebook 05: CNN + Bi-LSTM pada 78 Subjek, Bahan Presentasi

Dokumen ini untuk tim yang akan mempresentasikan `notebooks/05_cnn_bilstm_sc78.ipynb`. Istilah mengikuti tabel istilah di notebook 01.

---

## 1. Pesan utama

1. **Skor gabungan.** Model dari notebook 04 diuji tanpa perubahan pada seluruh 78 subjek (usia 25–101 tahun). Model lebih baik daripada *baseline* pada skor gabungan:

   | | Model | *Baseline* |
   |---|---|---|
   | Macro-F1 | 78,0 ± 0,6 | 76,5 |
   | κ | 0,774 | 0,739 |
   | Akurasi | 83,6 | 80,7 |

2. **Per subjek.** Model lebih baik untuk subjek tipikal: 48 dari 78 subjek, median +1,6 poin, Wilcoxon p = 0,005. Namun rerata keunggulannya (+1,06, interval 95% [−0,26; +2,29]) belum terbukti di atas nol, karena model gagal parah pada tiga subjek. Ketiga kegagalan muncul di semua *seed*.
3. **N1 dan N3.** Model lebih jarang menangkap N1 dan N3 daripada *baseline*. Penyebab utamanya bobot kelas pada *baseline*, bukan kelemahan arsitektur. *Precision* model untuk kedua tahap justru lebih tinggi.
4. **Usia, EMG, dan penurunan skor.**
   - Keunggulan model tidak bergantung pada usia.
   - Keunggulan model tidak hilang pada rekaman dengan EMG bermasalah.
   - Turunnya skor dari 80,1 (20 subjek muda) ke 78,0 berasal dari data uji yang lebih sulit, bukan dari model yang memburuk.

## 2. Posisi dalam proyek

Notebook 04 hanya menguji 20 subjek berusia 25–34 tahun. Notebook 05 adalah patokan utama proyek: seluruh Sleep Cassette, dengan 10 *fold* yang sama persis dengan *baseline*.

**Pengaturan model dibekukan.** Semua pengaturan disalin dari notebook 04 dan tidak disetel ulang. Alasannya, 20 subjek muda itu juga termasuk dalam 78 subjek. Kalau pengaturan dipilih ulang setelah melihat hasil mereka, angka di sini akan terlalu optimistis.

**Batas yang harus diakui.** Untuk 20 subjek muda, angka di sini tidak sepenuhnya bersih, karena model dikembangkan sambil melihat hasil mereka. Untuk 58 subjek lainnya, angkanya bersih. Data yang benar-benar baru baru diuji di Sleep Telemetry.

## 3. Alur presentasi

| Urutan | Yang ditampilkan | Yang diucapkan | Angka kunci |
|---|---|---|---|
| 1 | Bukti *fold* identik (bagian C) | Kedua model diuji pada orang yang sama, jadi perbandingan per subjek sah | MF1 *baseline* per *fold* dihitung ulang = 76,4 ± 2,4 |
| 2 | Tabel hasil (bagian F) | Model unggul pada skor gabungan | 78,0 vs 76,5 |
| 3 | Grafik per subjek (G1) | Lebih baik untuk kebanyakan orang, tetapi ada tiga kegagalan besar | 48 dari 78; p = 0,005; interval [−0,26; +2,29] |
| 4 | Matriks kebingungan dan porsi tebakan (G2) | Sensitivitas N1/N3 lebih rendah karena *baseline* memakai bobot kelas | Porsi tebakan N1: *baseline* 14,9%, model 10,2%, sebenarnya 11,0% |
| 5 (opsional) | Tabel usia (G3) dan EMG (G4) | Keunggulan tidak bergantung pada usia atau kualitas EMG | Korelasi usia–selisih 0,07 |

G5 (efek data latih) dan G6 (kurva latih) disiapkan untuk menjawab pertanyaan.

## 4. Penjelasan per bagian

### A. Pengaturan

Pengaturannya sama dengan notebook 04: *cross-entropy* tanpa bobot kelas, jendela 20 *epoch*, dan penghentian dini berdasarkan Macro-F1 validasi.

Kode model dan pelatihan diambil dari modul bersama `src/sleep_dl.py`, bukan disalin ke notebook. Modul ini diuji (`tests/test_sleep_dl.py`) menghasilkan angka identik dengan kode notebook 04. Kode yang disalin antar-notebook pernah menjadi sumber kesalahan pada versi pertama model.

### B. Data

**Cara memuat.** EEG dan EOG (3,5 GB) dibaca langsung dari disk saat dibutuhkan (*memory-mapping*), tidak disalin ke RAM. Isinya identik dengan memuat biasa, dan pembacaannya cukup cepat (bagian D).

**Komposisi per kelompok usia.**

| Usia | Subjek | Rekaman | *Epoch* berlabel | N1 (%) | N3 (%) | Rekaman EMG bermasalah |
|---|---|---|---|---|---|---|
| 25–39 | 20 | 39 | 42.307 | 6,6 | 13,5 | 2 |
| 40–59 | 21 | 41 | 50.896 | 10,3 | 5,5 | 2 |
| 60–79 | 21 | 41 | 49.428 | 13,7 | 6,2 | 2 |
| 80+ | 16 | 32 | 52.838 | 12,6 | 2,8 | 11 |

**Makna.**
- Kelompok 25–39 tahun berisi tepat 20 subjek dari notebook 04. Jadi notebook 04 hanya melihat seperempat populasi, yaitu yang termuda.
- Batas kelompok usia dipilih agar setiap kelompok berisi 16–21 subjek. Ini bukan pembagian klinis.

### C. Pembagian *fold*

**Cara kerja.** Langkah notebook 03 diulang: subjek dibagi menjadi 10 kelompok, dan subjek validasi dipilih dengan *seed* yang sama.

Kesamaan dengan *baseline* dibuktikan dengan dua cara:
- Setiap subjek diuji tepat sekali, dan subjek validasi tidak pernah merangkap subjek uji.
- Macro-F1 *baseline* dihitung ulang per *fold* dengan pembagian baru. Hasilnya harus sama dengan angka yang disimpan notebook 03, yaitu 76,4 ± 2,4. Kalau satu subjek saja berpindah *fold*, angka ini hampir pasti berubah. Cara ini dipakai karena notebook 03 tidak menyimpan daftar *fold*-nya.

**Keluaran.**
- Setiap *fold* berisi 7–8 subjek uji, 7–8 subjek validasi, dan 63 subjek latih.
- Macro-F1 *baseline* per *fold* berkisar 72,1–80,4.

**Makna.** Rentang 8 poin antar-*fold* jauh lebih besar daripada selisih antar-model yang diukur (1–2 poin). Karena itu perbandingan harus berpasangan per subjek.

**Kenapa 10 *fold* dan bukan satu subjek per *fold*?** Satu subjek per *fold* berarti 78 pelatihan per *seed*, sekitar 7 jam. Dengan 10 *fold*, setiap model tetap dilatih dengan 63 subjek, dan pembagiannya sama dengan *baseline*.

### D. Pemeriksaan sebelum pelatihan

- Jumlah parameter sama dengan notebook 04: 1.089.477.
- Model diuji pada rekaman subjek tertua (101 tahun), yang paling berbeda dari data notebook 04. Keluarannya tidak mengandung nilai rusak.
- Satu *batch* dibaca dari disk dalam sekitar 39 ms, dan satu *fold* selesai dilatih dalam sekitar 5 menit.

### E. Pelatihan

| *Seed* | Macro-F1 uji | κ | Rerata Macro-F1 validasi | Waktu |
|---|---|---|---|---|
| 0 | 77,4 | 0,769 | 77,9 | 54 menit |
| 1 | 78,5 | 0,777 | 78,1 | 59 menit |
| 2 | 78,3 | 0,778 | 77,9 | 55 menit |

**Pelatihan yang terputus.** Hasil setiap *fold* langsung disimpan, sehingga pelatihan yang terputus bisa dilanjutkan dari *fold* terakhir yang selesai.
- Fitur ini benar-benar terpakai. Pelatihan dihentikan sistem karena RAM laptop penuh, setelah *seed* 0 dan satu *fold* *seed* 1 selesai, lalu dilanjutkan tanpa mengulang.
- Saat itu ditemukan satu *bug*: di Windows, berkas sementara tidak bisa dihapus karena masih terbuka. *Bug* ini sudah diperbaiki, dan hasil *seed* 1 diperiksa identik dengan isi berkas sementaranya.

### F. Hasil keseluruhan

| | Model | *Baseline* | Selisih |
|---|---|---|---|
| Akurasi | 83,6 ± 0,4 | 80,7 | +3,0 |
| Macro-F1 | 78,0 ± 0,6 | 76,5 | +1,6 |
| κ | 0,774 ± 0,005 | 0,739 | +0,035 |
| F1 Wake | 93,2 | 91,8 | +1,4 |
| F1 N1 | 52,7 | 51,0 | +1,7 |
| F1 N2 | 85,0 | 81,5 | +3,6 |
| F1 N3 | 73,7 ± 1,8 | 74,6 | −0,9 |
| F1 REM | 85,6 | 83,5 | +2,1 |

**Makna.**
- Selisih Macro-F1 (+1,6) sekitar 2,5 kali simpangan baku antar-*seed*, jadi bukan kebetulan pelatihan. Namun selisih ini masih kecil dibandingkan simpangan baku antar-*fold* (3,1 untuk model, 2,4 untuk *baseline*).
- Keunggulan terbesar ada di N2 dan REM, dua tahap yang paling terbantu oleh konteks urutan.
- Model lebih tidak stabil antar-*fold* daripada *baseline*.

### G1. Perbandingan per subjek

**Cara kerja.**
- Macro-F1 dihitung per subjek, hanya dari tahap yang dimiliki subjek itu. Untuk model, nilainya dirata-ratakan atas 3 *seed*.
- Selisih model dan *baseline* dari 78 subjek diuji dengan dua cara:
  - interval *bootstrap* untuk rerata, dengan 10.000 pengambilan ulang;
  - uji Wilcoxon untuk peringkat.

**Keluaran.**
- Rerata selisih +1,06 [−0,26; +2,29]; median +1,6.
- Model lebih baik pada 48 dari 78 subjek; Wilcoxon p = 0,0052.
- Kuartil selisih adalah −0,5 dan +3,6.

**Tiga kegagalan besar.** Polanya sama pada ketiga *seed*:

| Subjek | Usia | *Baseline* | Model | Pola kesalahan model |
|---|---|---|---|---|
| 36 | 51 | 67,1 | 42,1 | REM sama sekali tidak dikenali (68 *epoch* terbaca N2/N1); 43 dari 48 *epoch* N3 terbaca N2 |
| 17 | 31 | 69,5 | 54,1 | Wake dibaca REM dan N1 (pada *seed* 0: 370 dan 227 dari 865 *epoch* Wake); N3 terbaca N2 |
| 25 | 56 | 80,0 | 64,7 | N3 tidak pernah ditebak (120 dari 120 *epoch* terbaca N2) |

Sebaliknya, ada subjek yang gagal pada *baseline* tetapi tidak pada model. Subjek 57 (66 tahun) mendapat 40,3 pada *baseline* dan 62,9 pada model.

**Kenapa dua uji tampak bertentangan?**
- Wilcoxon memakai peringkat. Tiga kegagalan besar hanya dihitung sebagai tiga peringkat negatif, sehingga hasilnya bermakna.
- *Bootstrap* memakai besar selisih. Tiga selisih −15 sampai −25 poin cukup untuk menarik rerata turun. Tanpa empat subjek terburuk, reratanya sekitar +1,9.
- Kedua uji benar. Keduanya menggambarkan model yang membaik untuk kebanyakan orang, tetapi sesekali gagal parah.

**Kalimat yang aman untuk presentasi.** "Model lebih baik daripada *baseline* untuk subjek tipikal (48 dari 78, p = 0,005). Namun rerata keunggulannya per subjek belum terbukti, karena model gagal parah pada sekitar 4% subjek."

Dalam pemakaian klinis, kegagalan seperti ini lebih penting daripada rerata: satu malam dengan REM yang hilang menghasilkan laporan yang salah. Penyebabnya belum diselidiki.

### G2. Jenis kesalahan per tahap

**Sensitivitas (%).**

| | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| *Baseline* | 89,3 | 60,0 | 76,9 | 84,6 | 83,8 |
| Model | 92,0 | 50,7 | 86,8 | 73,2 | 86,4 |
| Selisih | +2,7 | −9,3 | +10,0 | −11,4 | +2,6 |

**Porsi tebakan dan *precision*.** Tabel ini dihitung terpisah dari berkas prediksi dan tidak ada di notebook.

| | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| Porsi sebenarnya (%) | 33,7 | 11,0 | 35,4 | 6,7 | 13,2 |
| Porsi tebakan *baseline* (%) | 31,9 | 14,9 | 31,4 | 8,5 | 13,3 |
| Porsi tebakan model (%) | 32,9 | 10,2 | 36,9 | 6,6 | 13,5 |
| *Precision* *baseline* | 94,4 | 44,3 | 86,7 | 66,7 | 83,2 |
| *Precision* model | 94,3 | 54,8 | 83,3 | 74,2 | 84,9 |

**Makna.**
- *Baseline* dilatih dengan bobot kelas seimbang, sehingga lebih sering menebak N1 dan N3 daripada porsi sebenarnya. Akibatnya sensitivitasnya tinggi, tetapi banyak tebakannya salah.
- Model dilatih tanpa bobot kelas, dan porsi tebakannya hampir sama dengan porsi sebenarnya.
- F1 merangkum keduanya: F1-N1 naik 1,7 poin dan F1-N3 hanya turun 0,9.
- Jadi dari bagian ini tidak bisa disimpulkan bahwa arsitekturnya lemah pada N1 dan N3. Notebook 06 menguji langsung model yang sama dengan bobot kelas.
- Satu hal tetap nyata: 26% N3 terbaca sebagai N2, dan kesalahan ini terkumpul pada beberapa subjek (G1).

### G3. Kelompok usia

| Usia | Subjek | *Baseline* | Model | Selisih [95% CI] | Subjek lebih baik | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| 25–39 | 20 | 79,0 | 80,4 | +1,4 [−0,7; +3,0] | 14 | 0,012 | 0,048 |
| 40–59 | 21 | 73,2 | 72,7 | −0,5 [−3,8; +2,1] | 13 | 0,41 | 0,82 |
| 60–79 | 21 | 71,2 | 72,4 | +1,2 [−1,2; +4,0] | 11 | 0,84 | 0,84 |
| 80+ | 16 | 68,7 | 71,2 | +2,5 [+0,9; +4,3] | 10 | 0,025 | 0,075 |

Kolom Holm dihitung terpisah, karena notebook menampilkan p sebelum koreksi.

**Makna.**
- Kedua model kehilangan sekitar 10 poin dari kelompok termuda ke tertua. Korelasi usia dengan skor adalah −0,42 untuk *baseline* dan −0,36 untuk model.
- Keunggulan model tidak berubah dengan usia: korelasi usia dengan selisih hanya 0,07.
- Kelompok 40–59 tahun bernilai negatif karena memuat subjek 25 dan 36.
- Setelah koreksi Holm, hanya kelompok termuda yang bermakna. Namun kelompok itu dipakai selama pengembangan model, jadi tidak bisa dijadikan bukti.
- Analisis ini eksploratif.

### G4. Rekaman dengan EMG bermasalah

| | Rekaman | Macro-F1 *baseline* | Macro-F1 model | F1-REM *baseline* | F1-REM model |
|---|---|---|---|---|---|
| Semua | 153 | 76,5 | 78,0 | 83,5 | 85,6 |
| Tanpa tanda | 136 | 77,0 | 78,3 | 84,9 | 86,8 |
| Bertanda saja | 17 | 72,2 | 75,3 | 71,6 | 75,0 |

Karena kebanyakan rekaman bertanda milik subjek tua, perbandingan yang adil dilakukan di antara subjek berusia 60 tahun ke atas:

| Subjek 60+ | Subjek | *Baseline* | Model | Selisih |
|---|---|---|---|---|
| Tanpa tanda | 26 | 71,6 | 72,7 | +1,1 |
| Bertanda | 11 | 66,7 | 70,0 | +3,3 |

**Makna.**
- EMG yang buruk menurunkan F1-REM kedua model sebesar 11–13 poin.
- Model tidak lebih rentan, malah keunggulannya sedikit lebih besar pada rekaman bertanda. Dugaannya, model bisa mengandalkan EOG dan EEG untuk REM. Dugaan ini belum diuji.
- Membuang rekaman bertanda tidak mengubah kesimpulan, jadi rekaman itu tetap dipakai.
- Hasil ini hanya didukung 11 subjek.

### G5. Efek data latih pada 20 subjek muda

**Pertanyaan.** Apakah turunnya skor dari 80,1 ke 78,0 berarti model memburuk? Untuk menjawabnya, 20 subjek muda yang sama dibandingkan dalam dua kondisi:
- dilatih dengan 17 subjek muda (notebook 04);
- dilatih dengan sekitar 63 subjek dari semua usia (notebook 05).

**Hasil.**
- Skor gabungan naik dari 80,1 ke 80,8.
- Per subjek: +1,15 [−0,42; +2,46]; 14 dari 20 subjek lebih baik; p = 0,030.
- Subjek 17 turun dari 63,8 ke 54,1. Pada *baseline*, subjek ini justru naik dari 65,3 ke 69,5.

**Makna.**
- Data latih yang lebih banyak dan beragam tidak merusak hasil pada subjek muda.
- Turunnya skor di SC-78 berasal dari subjek uji yang lebih sulit.
- Keterbatasannya: jumlah data, keragaman data, dan subjek validasi berubah bersamaan, sehingga efeknya tidak bisa dipisahkan.

### G6. Kurva pelatihan

**Hasil.**
- Putaran terbaik: median 5, rentang 1–18.
- Skor validasi sering naik-turun 2 poin atau lebih antar-putaran, kadang sampai sekitar 6 poin.

**Makna.**
- Data validasi 7–8 subjek memberi sinyal yang berisik, sehingga putaran yang terpilih sebagian ditentukan kebetulan.
- Laju belajar hampir belum turun saat titik terbaik tercapai, karena penjadwal diatur untuk 30 putaran.
- Ada dua kandidat perbaikan untuk model final:
  - merata-ratakan bobot beberapa putaran terakhir;
  - menghaluskan skor validasi.

  Keduanya tidak dikerjakan di sini agar pengaturan tetap beku.

## 5. Keterbatasan yang perlu diakui

- **Tiga kegagalan besar.** Penyebabnya belum diselidiki. Yang perlu diperiksa: sinyal mentah dan artefak, normalisasi per rekaman, dan fitur yang membantu *baseline* tetapi tidak dimiliki model.
- **20 subjek muda tidak sepenuhnya bersih**, karena dipakai selama pengembangan model.
- **Perbandingan dengan *baseline* tidak sepenuhnya setara.** *Baseline* dilatih dengan bobot kelas dan dijalankan satu kali, sedangkan model tanpa bobot kelas dan dengan tiga *seed*.
- **Analisis usia dan EMG eksploratif**, dengan kelompok kecil.

## 6. Antisipasi pertanyaan

**"Kenapa skor di 78 subjek lebih rendah daripada di 20 subjek?"**
Pada 20 subjek yang sama, model yang dilatih di sini justru lebih baik (80,8 vs 80,1). Rerata turun karena subjek tua lebih sulit, dan *baseline* pun turun dengan pola yang sama (79,2 ke 76,5).

**"Wilcoxon bermakna, tetapi interval *bootstrap* mencakup nol. Mana yang benar?"**
Keduanya benar, karena menjawab pertanyaan berbeda:
- apakah kebanyakan subjek membaik? Ya;
- apakah rerata selisihnya pasti di atas nol? Belum, karena ada tiga kegagalan parah.

Laporan yang jujur menyebutkan keduanya.

**"Kenapa sensitivitas N1 model lebih rendah, padahal N1 yang paling ingin diperbaiki?"**
*Baseline* memakai bobot kelas, sehingga lebih sering menebak N1, baik yang benar maupun yang salah. F1-N1 model tetap lebih tinggi. Kalau sensitivitas N1 yang diutamakan, bobot kelas bisa ditambahkan, dan dampaknya diukur di notebook 06.

**"Bisakah dibandingkan dengan makalah lain?"**
Hanya dengan hati-hati. Angka di literatur bergantung pada:
- berapa lama Wake yang disertakan;
- apakah *epoch* gerakan dibuang;
- jumlah subjek;
- cara membagi data.

Proyek ini memakai *trimming* 30 menit, mengecualikan *epoch* gerakan dan *epoch* yang tidak dinilai, dan membagi data per subjek.

**"Kenapa tidak menyetel ulang pengaturan untuk 78 subjek?"**
Satu-satunya data untuk menilai penyetelan itu adalah data uji ini sendiri. Penyetelan dengan data uji membuat angka akhir terlalu optimistis.

**"Apa yang terjadi pada tiga subjek yang gagal?"**
Polanya sudah diketahui: REM hilang, N3 hilang, atau Wake dibaca sebagai REM. Penyebabnya belum diselidiki. Ini dilaporkan sebagai keterbatasan, bukan disembunyikan.

## 7. Lampiran

**Kode utama** (di `src/sleep_dl.py`, dipanggil sebagai `sd.*`).

| Fungsi | Gunanya |
|---|---|
| `load_records` | Memuat 153 rekaman; EEG/EOG dibaca dari disk |
| `group_kfold_folds` | 10 *fold* per subjek, identik dengan notebook 03 |
| `run_cv` | Melatih dan menguji semua *fold*; hasil disimpan per *fold* dan bisa dilanjutkan |
| `paired` | *Bootstrap* dan Wilcoxon untuk selisih per subjek |

**Berkas.**
- Tabel: `results/tables/05_cnn_sc78_*.csv` (per *seed*, metrik, per subjek, usia, EMG, uji berpasangan).
- Gambar: `results/figures/05_cnn_sc78_*.png` (per subjek, matriks kebingungan, kurva latih).
- Prediksi:
  - per *seed*: `data/processed/cnn_sc78_runs/ce_seed{0,1,2}.npz`;
  - rerata tiga *seed*: `data/processed/cnn_sc78_predictions.npz`.

**Keputusan desain.** K7 di `docs/catatan/KEPUTUSAN_DESAIN.md`.
