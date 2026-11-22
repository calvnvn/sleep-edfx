# Notebook 03: *Baseline* Fitur Manual, Bahan Presentasi

Dokumen ini untuk tim yang akan mempresentasikan `notebooks/03_baseline.ipynb`. Istilah mengikuti tabel istilah di notebook 01.

---

## 1. Pesan utama

1. **Angka *baseline*.** Fitur manual yang digabungkan dengan *gradient boosting* mencapai:
   - SC-20: Macro-F1 79,2 dan κ 0,784;
   - SC-78: Macro-F1 76,5 dan κ 0,739.

   Ini batas bawah yang harus dilampaui model *deep learning*, dengan pembagian data yang sama persis.
2. **Sinyal dan konteks.** Setiap sinyal tambahan (EEG kedua, EOG, EMG) dan konteks dari *epoch* tetangga menaikkan Macro-F1, dari 68,4 dengan satu kanal EEG sampai 79,2 dengan semuanya. Temuan ini menjadi dasar rancangan model berikutnya.
3. **Kebocoran data.** Kalau data dibagi acak per *epoch*, Macro-F1 naik palsu sebesar 7 poin. Karena itu seluruh evaluasi di proyek ini dilakukan per subjek.
4. **Tahap dan subjek tersulit.** N1 adalah tahap tersulit (F1 sekitar 50), dan subjek tua adalah kelompok tersulit (Macro-F1 79 pada usia 25–39 tahun, 69 pada usia 80 tahun ke atas).

## 2. Posisi dalam proyek

*Baseline* punya tiga fungsi:
- **Batas bawah.** Model yang lebih rumit harus terbukti lebih baik dari ini.
- **Uji kewarasan data.** Kalau model sederhana pun gagal, kemungkinan ada masalah pada data atau pembagiannya.
- **Pembanding untuk analisis sinyal dan kebocoran.** Analisis yang sama diulang pada model *deep learning* di notebook 06.

| | *Baseline* (notebook 03) | *Deep learning* (notebook 04–06) |
|---|---|---|
| Yang menentukan ciri | Peneliti, berdasarkan ilmu tidur | Jaringan saraf, dari data |
| Masukan | 33 angka per *epoch* (165 dengan konteks) | Sinyal yang sudah diolah di notebook 02 |
| Konteks | Fitur dua *epoch* tetangga ditempelkan | Bi-LSTM membaca urutan 20 *epoch* |
| Bobot kelas | Seimbang | Tidak (model referensi) |
| Waktu | Sekitar 25 menit di CPU | Jam, di GPU |

## 3. Alur presentasi

| Urutan | Yang ditampilkan | Yang diucapkan | Angka kunci |
|---|---|---|---|
| 1 | Daftar fitur (bagian B) dan grafik fitur per tahap (bagian D) | Fitur diturunkan dari ciri yang dipakai pakar, dan perilakunya sesuai fisiologi | Gelombang lambat tertinggi saat N3, EMG terendah saat REM |
| 2 | Tabel hasil dan matriks kebingungan (bagian G) | Angka batas bawah; N1 paling sulit | Macro-F1 79,2 / 76,5; N1 benar 55–60% |
| 3 | Tabel sumbangan sinyal (H1) | Setiap sinyal dan konteks menambah skor | 68,4 → 79,2 |
| 4 | Tabel kebocoran (H2) | Kesalahan pembagian data bisa menambah 7 poin palsu | 79,2 vs 86,2 |
| 5 | Grafik skor per usia (H4) | Subjek tua lebih sulit; hasil harus dilaporkan per kelompok usia | 79,0 → 68,7 |

## 4. Penjelasan per bagian

### B. Definisi fitur

Fitur diturunkan dari dua sumber:
- **Ciri yang dipakai pakar menurut aturan AASM:**
  - alfa saat terjaga;
  - theta saat N1;
  - *spindle* saat N2;
  - gelombang lambat saat N3;
  - gerakan mata cepat dan otot lumpuh saat REM.
- **Rumus pengolahan sinyal yang baku:** spektrum Welch, parameter Hjorth, dan kurtosis.

| Kanal | Jumlah | Fitur |
|---|---|---|
| Setiap kanal EEG | 12 | Daya total; daya relatif enam pita (0,5–2, 2–4, 4–8, 8–12, 12–15, 15–30 Hz); rasio theta/alfa dan delta/beta; dua parameter Hjorth; kurtosis |
| EOG | 5 | Daya gerakan lambat (0,3–2 Hz) dan cepat (2–8 Hz), rasio keduanya, amplitudo puncak-ke-puncak, kurtosis |
| EMG | 4 | Rerata, simpangan baku, minimum, maksimum dari 30 nilai per *epoch* |

**Cara kerja fitur utama.**
- **Daya pita:** luas di bawah kurva spektrum pada rentang frekuensi itu. Spektrum dihitung dengan metode Welch, yaitu jendela 4 detik dengan tumpang tindih 50%.
- **Daya relatif:** daya pita dibagi daya total, sehingga tidak bergantung pada skala rekaman. Logaritma dipakai karena nilai daya tersebar sangat lebar.
- **Parameter Hjorth:**
  - *mobility* memperkirakan frekuensi rata-rata, dan nilainya rendah saat gelombang lambat;
  - *complexity* mengukur seberapa berubah-ubah frekuensinya.
- **Kurtosis:** tinggi kalau ada lonjakan sesaat, misalnya kedipan mata atau *K-complex*.

### C. Ekstraksi

Hasilnya 195.767 *epoch* × 33 fitur, selesai dalam 2–4 menit. Ada 298 *epoch* berlabel −1 yang ikut disimpan, karena fiturnya masih berguna sebagai konteks bagi *epoch* tetangga.

### D. Pemeriksaan fitur per tahap

| Fitur | Pola | Makna |
|---|---|---|
| Daya gelombang lambat (Fpz-Cz) | Tertinggi saat N3 | Ciri tidur dalam |
| Daya alfa (Pz-Oz) | Tertinggi saat Wake | Irama alfa, terkuat di belakang kepala |
| Hjorth *mobility* | Terendah saat N3 | Frekuensi rata-rata turun |
| Daya sigma (Fpz-Cz) | **Tidak** tertinggi saat N2 | Lihat catatan di bawah |
| Amplitudo EOG | Tertinggi saat Wake dan REM | Kedipan dan gerakan mata cepat |
| EMG | Turun dari Wake sampai REM | Otot lumpuh saat REM |

**Catatan sigma.**
- Pita 12–15 Hz adalah pita *spindle*, ciri N2, jadi dayanya diharapkan paling tinggi saat N2.
- Kenyataannya, median saat N1 sedikit lebih tinggi daripada N2, dan rentang keduanya tumpang tindih. Sigma tetap memisahkan N3, karena nilainya jauh lebih rendah saat N3.
- Ada tiga dugaan penyebab, semuanya belum diuji:
  - *spindle* hanya berlangsung sekitar 0,5–2 detik, sehingga tenggelam dalam rata-rata 30 detik;
  - daya relatif dibagi daya total, dan daya total ikut berubah antar-tahap;
  - batas pita 12–15 Hz lebih sempit daripada yang dipakai sebagian penulis (11–16 Hz).

### E. Konteks dari *epoch* tetangga

Fitur dua *epoch* sebelum dan dua sesudahnya ditempelkan ke setiap baris, sehingga setiap baris berisi 165 kolom.
- Di tepi rekaman, tetangga yang tidak ada diisi kosong, bukan nol. Nol adalah nilai fitur yang sah dan akan menyesatkan model.
- Jumlah nilai kosong tepat 30.294, yaitu 153 rekaman × 6 posisi tepi × 33 fitur. Angka ini membuktikan tidak ada tetangga yang terambil dari rekaman lain.

### F. Model dan cara validasi

**Model.**
- *Histogram gradient boosting* dari scikit-learn: laju belajar 0,1, maksimum 31 daun per pohon, dan bobot kelas seimbang.
- Cara kerjanya: banyak pohon keputusan kecil dibangun berurutan, dan setiap pohon memperbaiki kesalahan pohon-pohon sebelumnya.

**Cara validasi.**
1. Data dibagi per subjek.
2. Di setiap *fold*, sekitar 10% subjek latih disisihkan sebagai data validasi.
3. Pelatihan berhenti kalau galat validasi tidak membaik selama 20 iterasi, dengan batas maksimum 500 iterasi.
4. Data uji tidak pernah dipakai untuk keputusan apa pun.

### G. Hasil utama

| | Akurasi | Macro-F1 | F1 Wake | F1 N1 | F1 N2 | F1 N3 | F1 REM | κ |
|---|---|---|---|---|---|---|---|---|
| SC-20 (20 *fold*) | 84,1 | 79,2 | 89,8 | 49,3 | 86,8 | 83,6 | 86,5 | 0,784 |
| SC-78 (10 *fold*) | 80,7 | 76,5 | 91,8 | 51,0 | 81,5 | 74,6 | 83,5 | 0,739 |

**Variasi per *fold*.** Rerata Macro-F1 per *fold* adalah 78,6 ± 5,8 (SC-20) dan 76,4 ± 2,4 (SC-78). Sebaran SC-20 lebih lebar karena setiap *fold* hanya berisi satu orang.

**Matriks kebingungan.**
- N1 hanya benar 55–60%. Sisanya tertukar dengan N2, Wake, dan REM, yaitu tahap yang secara fisiologis bertetangga dengan N1.
- 13–14% N3 terbaca sebagai N2.
- Pada SC-78, 12% N2 terbaca sebagai N1. Kemungkinan ini akibat bobot kelas yang mendorong model lebih sering menebak N1.

**Perbandingan dengan literatur.** Skor ini dekat dengan beberapa model *deep learning* awal, misalnya DeepSleepNet dengan Macro-F1 76,9 pada SC-20. Namun protokolnya tidak persis sama, jadi perbandingan langsung harus hati-hati.

### H1. Sumbangan tiap sinyal (SC-20)

| Masukan | Jumlah fitur | Macro-F1 | F1-N1 | F1-REM |
|---|---|---|---|---|
| Fpz-Cz | 12 | 68,4 | 29,3 | 68,4 |
| + Pz-Oz | 24 | 72,5 | 34,6 | 73,9 |
| + EOG | 29 | 75,1 | 41,7 | 79,1 |
| + EMG | 33 | 76,6 | 43,7 | 82,9 |
| + konteks | 165 | 79,2 | 49,3 | 86,5 |

**Makna.**
- Tahap yang paling terbantu oleh setiap tambahan sesuai dengan fisiologinya:
  - Pz-Oz membantu Wake, lewat irama alfa;
  - EOG dan EMG membantu REM;
  - konteks membantu N1 dan REM.
- Penambahan dilakukan berurutan (kumulatif), sehingga besar kenaikan tiap langkah bergantung pada urutannya.
- Hasil berasal dari satu *seed*.

### H2. Kebocoran data (SC-20)

| Pembagian | Akurasi | Macro-F1 | F1-N1 | κ |
|---|---|---|---|---|
| Per subjek (benar) | 84,1 | 79,2 | 49,3 | 0,784 |
| Acak per *epoch* (bocor) | 90,0 | 86,2 | 63,6 | 0,863 |

**Dua jalur kebocoran.**
- **Model mengenali orangnya.** Model belajar ciri sinyal khas tiap orang, lalu memakainya untuk menebak.
- **Tetangga yang hampir kembar.** Kalau *epoch* 500 masuk data uji, *epoch* 499 dan 501 kemungkinan besar ada di data latih. Isi keduanya hampir sama dan labelnya biasanya sama. Lewat fitur konteks, baris *epoch* 501 bahkan memuat fitur *epoch* 500, lengkap dengan label *epoch* 501.

Jalur kedua adalah penalaran yang belum diuji terpisah. Selisih 7 poin ini juga konsisten dengan percobaan awal proyek sebelum data diolah (73,7 vs 81,0).

### H3. Fitur yang paling diandalkan (SC-78)

**Cara kerja.** Satu kelompok kolom pada data uji diacak antar-*epoch*, lalu diukur seberapa besar Macro-F1 turun. Teknik ini disebut *permutation importance*.

| Sinyal | Penurunan Macro-F1 |
|---|---|
| EEG Fpz-Cz | 21,6 |
| EEG Pz-Oz | 19,8 |
| EOG | 11,2 |
| EMG | 1,9 |

Fitur tunggal teratas adalah daya total Fpz-Cz (5,5 poin), disusul kurtosis Fpz-Cz, kurtosis EOG, dan daya delta 2–4 Hz Pz-Oz.

**Kenapa EMG tampak kecil padahal menambah skor di H1?** Kedua analisis menjawab pertanyaan yang berbeda:
- H1 mengukur tambahan skor kalau EMG ditambahkan;
- H3 mengukur kerugian kalau EMG dirusak sementara sinyal lain tetap ada.

Sebagian informasi EMG kemungkinan juga terkandung di EOG dan EEG, sehingga model bisa mengandalkan sinyal lain saat EMG dirusak. Tafsiran ini belum dibuktikan.

### H4. Skor per subjek menurut usia (SC-78)

| Usia | Subjek | Rerata Macro-F1 | Simpangan baku |
|---|---|---|---|
| 25–39 | 20 | 79,0 | 4,7 |
| 40–59 | 21 | 73,2 | 6,7 |
| 60–79 | 21 | 71,2 | 10,4 |
| 80+ | 16 | 68,7 | 8,3 |

**Hasil.**
- Korelasi usia dengan skor adalah −0,43.
- Subjek dengan EMG bermasalah rata-rata 68,8, dibandingkan 74,3 untuk subjek lainnya. Namun kebanyakan dari mereka berusia lanjut, sehingga efek EMG dan efek usia belum bisa dipisahkan di sini. Keduanya dipisahkan di notebook 05.

Macro-F1 per subjek dihitung hanya dari tahap yang ada pada subjek itu. Dengan cara ini, subjek yang tidak punya N3 tidak otomatis mendapat F1 nol untuk N3.

## 5. Keterbatasan yang perlu diakui

- **Satu *seed*.** Analisis di bagian H berasal dari satu *seed* tanpa interval kepercayaan, jadi selisih yang kecil belum pasti.
- **Batas pita frekuensi.** Batas yang dipakai adalah konvensi dan belum dibandingkan dengan alternatif.
- **Bobot kelas.** *Baseline* memakai bobot kelas seimbang, sedangkan model referensi *deep learning* tidak. Perbedaan ini memengaruhi perbandingan untuk N1 dan N3, dan diuji di notebook 06.
- **Penjelasan penyebab.** Penjelasan seperti tetangga kembar, tumpang tindih informasi EMG, dan efek usia adalah tafsiran yang konsisten dengan data, bukan hasil uji khusus.

## 6. Antisipasi pertanyaan

**"Kenapa *gradient boosting*, bukan *random forest* atau SVM?"**
*Gradient boosting* menangani nilai kosong dengan sendirinya, sehingga *epoch* di tepi rekaman tidak perlu diisi nilai palsu. Metode ini juga cepat untuk 195 ribu baris, dan termasuk metode terkuat untuk data tabel. Tujuan *baseline* adalah menjadi pembanding yang kuat, bukan yang lemah.

**"Apakah *baseline* ini terlalu kuat atau terlalu lemah?"**
Skornya setara dengan beberapa model *deep learning* awal di literatur. Jadi ia bukan pembanding yang sengaja dibuat lemah supaya model kita tampak bagus.

**"Kenapa evaluasi per subjek begitu penting?"**
Pada data kami, pembagian acak per *epoch* menaikkan Macro-F1 dari 79,2 menjadi 86,2 dengan model yang sama persis. Dalam pemakaian nyata, model selalu menghadapi orang baru.

**"Kenapa N1 paling sulit?"**
- N1 adalah tahap peralihan tanpa ciri khas sendiri.
- Durasinya singkat dan sering berganti ke tahap lain.
- Jumlahnya sedikit, hanya 11% data.
- Pakar sendiri hanya sepakat sekitar 63% untuk N1 (Rosenberg & Van Hout, 2013).

**"Kalau EMG hanya menyumbang 1,9 poin, kenapa tetap dipakai?"**
Tanpa EMG, F1-REM lebih rendah 3,8 poin (79,1 vs 82,9 pada H1). EMG adalah pembeda REM dari Wake yang paling langsung secara fisiologis.

## 7. Lampiran: membaca kode

| Potongan kode | Fungsinya |
|---|---|
| `signal.welch(x, fs=FS, nperseg=4 * FS)` | Spektrum daya dengan jendela 4 detik |
| `np.roll(X, k, axis=0)` | Menggeser tabel untuk mengambil fitur tetangga |
| `HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_leaf_nodes=31, ...)` | Model *baseline*, dengan bobot kelas seimbang dan penghentian dini |
| `LeaveOneGroupOut`, `GroupKFold` | Pembagian *fold* per subjek: satu subjek per *fold* (SC-20), atau 10 kelompok (SC-78) |
| `GroupShuffleSplit` | Memilih subjek validasi dari data latih |
| `f1_score(..., average="macro")`, `cohen_kappa_score` | Metrik utama |

**Berkas.**
- Fitur: `data/processed/features_sc.npz`.
- Prediksi: `data/processed/baseline_predictions.npz`.
- Tabel: `results/tables/03_baseline_*.csv`.

**Keputusan desain.** K5 (rancangan *baseline*) dan K3 (percobaan kebocoran) di `docs/catatan/KEPUTUSAN_DESAIN.md`.
