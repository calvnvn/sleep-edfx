# Notebook 01: Eksplorasi Data, Bahan Presentasi

Dokumen ini untuk tim yang akan mempresentasikan `notebooks/01_eda.ipynb`. Isinya:
- pesan yang harus sampai ke penguji;
- sel yang ditampilkan dan apa yang diucapkan;
- penjelasan setiap sel;
- jawaban untuk pertanyaan yang mungkin muncul.

Semua angka berasal dari keluaran notebook. Istilah teknis mengikuti tabel istilah di awal notebook 01.

---

## 1. Pesan utama

Dalam satu menit, penguji perlu menangkap empat hal:

1. **Data.** Sleep-EDF Expanded berisi 197 rekaman dari 100 orang. Yang dipakai untuk melatih dan menguji model adalah Sleep Cassette (SC): 153 rekaman dari 78 orang sehat berusia 25–101 tahun. Sleep Telemetry (ST), 44 rekaman, disimpan untuk uji eksternal.
2. ***Trimming*.** Rekaman SC hampir sehari penuh, sehingga 68,8% *epoch* adalah keadaan terjaga. Setelah hanya disisakan periode tidur ditambah 30 menit di kedua sisi, porsinya turun menjadi 33,7%.
3. **Ketimpangan kelas.** Setelah *trimming* pun, N1 (11,0%) dan N3 (6,7%) tetap jarang. Karena itu metrik utama proyek adalah Macro-F1, bukan akurasi.
4. **Usia.** Komposisi tidur berubah besar menurut usia: N3 turun dari 13,5% (25–39 tahun) ke 2,8% (80 tahun ke atas). Hasil pada subjek muda tidak boleh dianggap berlaku untuk semua usia.

## 2. Posisi dalam proyek

```
01 Eksplorasi  →  02 Pra-pemrosesan  →  03 Baseline  →  04–05 Model DL  →  06 Ablasi  →  Uji eksternal ST
(memahami)        (mengubah data)       (pembanding)     (model utama)      (komponen)    (generalisasi)
```

Notebook 01 tidak membaca sinyal dan tidak mengubah data. Ia hanya membaca ringkasan per rekaman dari `results/eda/eda_records.json`. Ringkasan itu dibuat sekali oleh `scripts/edfx_eda.py` dari *header* berkas sinyal dan berkas label. Pemisahan ini sengaja dibuat: membaca 8 GB berkas mentah lambat dan cukup dilakukan sekali, sedangkan notebook ini bisa dijalankan ulang dalam hitungan detik.

Kata *trimming* di notebook ini hanya berarti **menghitung** berapa *epoch* yang akan tersisa. Pemangkasan yang sebenarnya, pada sinyal dan label, dilakukan di notebook 02.

## 3. Alur presentasi

| Urutan | Yang ditampilkan | Yang diucapkan | Angka kunci |
|---|---|---|---|
| 1 | Tabel ringkasan (bagian B) | Isi dataset dan alasan memakai SC sebagai data utama | 153 rekaman SC, 78 subjek; ST 44 rekaman |
| 2 | Grafik sebelum/sesudah *trimming* (bagian C) | Tanpa *trimming*, model yang selalu menebak "terjaga" sudah benar 69% | Wake 68,8% → 33,7% |
| 3 | Grafik komposisi menurut usia (bagian F) | Tidur dalam menyusut dengan usia; ini alasan patokan utama memakai 78 subjek, bukan hanya 20 subjek muda | N3 13,5% → 2,8% |
| 4 (opsional) | Grafik menit per tahap (bagian D) | Variasi antar-orang besar, terutama untuk Wake dan N3 | Median Wake 140 menit, rerata 216 |

Grafik durasi rekaman (bagian E) cukup disebut secara lisan: SC sekitar 23 jam dan ST sekitar 9 jam.

## 4. Penjelasan per sel

### A. Persiapan (2 sel)

**Tujuan.** Memuat pustaka, menetapkan urutan tahap (Wake, N1, N2, N3, REM), dan menetapkan warna tetap untuk setiap tahap. Warna yang sama dipakai di semua notebook supaya grafik mudah dibandingkan.

**Cara kerja.**
- Folder proyek dicari otomatis, dengan naik dari folder notebook sampai menemukan `results/eda`.
- Ringkasan 197 rekaman dimuat menjadi tabel.
- Daftar jumlah *epoch* per tahap, yang tersimpan sebagai satu daftar per rekaman, dipecah menjadi kolom terpisah.
- Data dibagi menjadi tiga kelompok: SC-78 (153 rekaman), SC-20 (subjek 0–19, 39 rekaman), dan ST (44 rekaman).

**Keluaran.** Tabel berukuran 197 baris × 29 kolom.

### A. Pemeriksaan angka (1 sel)

**Tujuan.** Mencocokkan jumlah rekaman dan jumlah *epoch* per tahap dengan angka yang dihitung secara terpisah. Kalau tidak cocok, notebook berhenti dengan galat.

**Mengapa penting.** Kesalahan paling berbahaya dalam analisis data adalah yang diam: grafik tetap tampil normal walaupun ada kolom yang tertukar. Pemeriksaan otomatis membuat kesalahan seperti itu langsung terlihat.

**Keluaran.** "All checks passed".

### B. Ringkasan dataset

**Keluaran.**

| | SC-78 | SC-20 | ST | Semua |
|---|---|---|---|---|
| Rekaman | 153 | 39 | 44 | 197 |
| Subjek | 78 | 20 | 22 | 100 |
| Usia (tahun) | 25–101 | 25–34 | 18–78 | 18–101 |
| Rerata durasi (jam) | 22,7 | 22,9 | 8,6 | 19,5 |
| *Epoch* berlabel setelah *trimming* | 195.469 | 42.307 | 42.467 | 237.936 |

**Makna.**
- SC-20 hanya berisi orang berusia 25–34 tahun. SC-20 berguna untuk percobaan cepat, tetapi tidak mewakili semua usia.
- Rekaman SC hampir sehari penuh karena alat portabel dipasang sejak sore dan baru dilepas keesokan harinya.

### C. Distribusi tahap sebelum dan sesudah *trimming*

**Cara kerja.** Jumlah *epoch* per tahap dijumlahkan atas semua rekaman SC, lalu diubah menjadi persen. Ini dilakukan dua kali, sebelum dan sesudah *trimming*.

**Keluaran.**

| Tahap | Sebelum (%) | Sesudah (%) |
|---|---|---|
| Wake | 68,8 | 33,7 |
| N1 | 5,2 | 11,0 |
| N2 | 16,7 | 35,4 |
| N3 | 3,1 | 6,7 |
| REM | 6,2 | 13,2 |

**Makna.**
- *Trimming* hanya membuang *epoch* terjaga yang jauh dari waktu tidur. Jumlah *epoch* tidur tidak berubah. Persentase tahap tidur naik semata karena jumlah Wake berkurang.
- Setelah *trimming*, perbandingan kelas terbesar (N2) dengan kelas terkecil (N3) sekitar 5:1. Ketimpangannya sedang: tidak ekstrem, tetapi cukup untuk membuat akurasi menyesatkan.

**Contoh satu rekaman (subjek 0, malam 1).**
- Rekaman mulai pukul 16.13. Subjek tertidur sekitar pukul 00.43 dan terbangun sekitar 06.44, sedangkan alat terus merekam sampai sekitar pukul 14.18.
- Dari 2.650 *epoch* (22,1 jam), *trimming* menyisakan 841 *epoch* (7,0 jam). Wake turun dari 1.997 menjadi 188 *epoch*:
  - 60 *epoch* sebelum tidur;
  - 68 *epoch* terjaga di tengah malam;
  - 60 *epoch* setelah tidur.

### D. Lama tiap tahap per malam

**Cara kerja.** Jumlah *epoch* dikali 0,5 menjadi menit. Rerata, simpangan baku, dan median dihitung per tahap atas 153 rekaman. Grafik menampilkan rerata sebagai batang, simpangan baku sebagai garis, dan setiap rekaman sebagai satu titik.

**Keluaran.**

| Tahap | Rerata (menit) | Simpangan baku | Median |
|---|---|---|---|
| Wake | 215,5 | 182,1 | 140,5 |
| N1 | 70,3 | 45,9 | 59,0 |
| N2 | 225,9 | 56,5 | 224,0 |
| N3 | 42,6 | 40,6 | 39,0 |
| REM | 84,4 | 31,5 | 84,5 |

**Makna.**
- N2 dan REM relatif stabil antar-rekaman, karena rerata dan mediannya hampir sama.
- Wake sangat miring: kebanyakan rekaman punya sekitar 140 menit, tetapi beberapa rekaman jauh lebih panjang dan menarik rerata ke atas. Untuk Wake, sebutkan median.
- Simpangan baku N3 hampir sama dengan reratanya. Variasi ini sebagian besar dijelaskan oleh usia (bagian F).

### E. Durasi rekaman SC vs ST

**Keluaran.** Median durasi SC 23,0 jam, ST 8,6 jam.

**Makna.** ST hanya merekam malam hari di rumah sakit, sehingga *trimming* hampir tidak mengubahnya. ST juga berbeda dalam hal subjek (sedikit sulit tidur), obat (separuh malam dengan temazepam), dan alat EMG. Karena itu ST tidak dicampur ke data latih. ST dipakai untuk menguji apakah model tetap bekerja pada data yang dikumpulkan dengan cara lain.

### F. Komposisi tahap menurut usia

**Cara kerja.** Subjek SC dikelompokkan menjadi 25–39, 40–59, 60–79, dan 80 tahun ke atas. Porsi tiap tahap dihitung per kelompok.

**Keluaran.**

| Usia | Rekaman | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|
| 25–39 | 39 | 19,6 | 6,6 | 42,1 | 13,5 | 18,2 |
| 40–59 | 41 | 33,2 | 10,3 | 37,7 | 5,5 | 13,4 |
| 60–79 | 41 | 31,2 | 13,7 | 35,9 | 6,2 | 13,0 |
| 80+ | 32 | 48,0 | 12,6 | 27,3 | 2,8 | 9,3 |

**Makna.**
- Seiring usia, tidur menjadi lebih ringan dan lebih sering terputus. Ini sesuai dengan meta-analisis Ohayon dkk. (2004).
- Akibatnya bagi model, orang tua punya lebih banyak N1, tahap yang paling sulit dikenali, dan lebih sedikit N3, tahap yang biasanya mudah. Skor model diperkirakan turun pada subjek tua, dan notebook 03 serta 05 mengonfirmasinya.
- Kelompok 25–39 tahun isinya tepat sama dengan SC-20.

## 5. Keterbatasan yang perlu diakui

- Statistik dihitung per rekaman, bukan per subjek. Subjek dengan dua malam mendapat bobot dua kali.
- Notebook ini tidak memeriksa kualitas sinyal. Masalah seperti EMG yang rusak baru ditemukan di notebook 02.
- Batas 30 menit untuk *trimming* mengikuti praktik umum di literatur (misalnya Supratak dkk., 2017), bukan hasil optimasi.

## 6. Antisipasi pertanyaan

**"Kenapa tidak memakai semua 197 rekaman untuk melatih?"**
ST berbeda dalam hal subjek, obat, tempat, dan alat. Kalau dicampur, kita tidak bisa lagi menguji apakah model bertahan pada data dari sumber lain. Selain itu, efek obat akan bercampur dengan pola tidur normal.

**"Kenapa *trimming* 30 menit, bukan nol atau satu jam?"**
Nol menghapus transisi dari terjaga ke tidur, padahal transisi itu bagian penting yang harus dikenali model. Batas yang terlalu panjang mengembalikan dominasi Wake. Batas 30 menit adalah konvensi yang dipakai banyak penelitian pada dataset ini, sehingga hasil kita bisa dibandingkan dengan penelitian lain.

**"Kenapa Macro-F1 dan bukan akurasi?"**
Akurasi bisa tinggi hanya dengan menebak kelas besar dengan benar. Macro-F1 memberi bobot sama pada kelima tahap, sehingga kegagalan pada N1 atau N3 langsung terlihat.

**"Apa dampak temuan usia pada rancangan eksperimen?"**
Patokan utama memakai seluruh 78 subjek (notebook 05), bukan hanya 20 subjek muda. Hasil juga dilaporkan per kelompok usia.

**"Apakah *trimming* mengubah label tidur?"**
Tidak. *Trimming* hanya membuang *epoch* terjaga di luar jendela. Pemeriksaan di notebook 02 memastikan jumlah *epoch* N1–REM tidak berubah.

## 7. Lampiran: membaca kode

| Potongan kode | Fungsinya |
|---|---|
| `df[df["group"] == "SC"]` | Menyaring baris, misalnya hanya rekaman SC |
| `pd.DataFrame(df.pop("full").tolist(), columns=LABELS)` | Memecah daftar jumlah *epoch* per rekaman menjadi tujuh kolom bernama |
| `assert kondisi` | Diam kalau kondisi benar; menghentikan notebook kalau salah |
| `d[cols].sum()`, lalu `/ total * 100` | Menjumlahkan per tahap atas semua rekaman, lalu mengubahnya menjadi persen |
| `minutes.agg(["mean", "std", "median"])` | Menghitung tiga ukuran sekaligus untuk setiap tahap |
| `pd.cut(age, bins=[24, 39, 59, 79, 120])` | Mengubah usia menjadi kelompok rentang |
| `groupby("age_group")[...].sum()` | Menjumlahkan *epoch* per kelompok usia |

**Prinsip grafik yang dipakai.**
- Judul setiap grafik menyatakan temuannya.
- Setiap tahap punya warna tetap di semua grafik.
- Warna aksen dipakai untuk fokus, dan abu-abu untuk konteks.
- Palet warna sudah diuji untuk pembaca buta warna. Kontras beberapa warna di bawah 3:1, dan ini dikompensasi dengan angka di dalam segmen grafik.

**Pustaka.** pandas, numpy, matplotlib.

**Berkas.**
- Gambar: `results/figures/01_*.png`.
- Tabel ringkasan: `results/tables/01_dataset_overview.csv`.

**Rujukan.**
- Kemp dkk. (2000), *IEEE TBME* 47(9).
- Ohayon dkk. (2004), *Sleep* 27(7).
- Supratak dkk. (2017), *IEEE TNSRE* 25(11).
- Rosenberg & Van Hout (2013), *J Clin Sleep Med* 9(1).
