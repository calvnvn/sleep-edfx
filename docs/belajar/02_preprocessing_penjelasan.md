# Notebook 02: Pra-pemrosesan, Bahan Presentasi

Dokumen ini untuk tim yang akan mempresentasikan `notebooks/02_preprocessing.ipynb`. Isinya:
- pesan utama;
- alur presentasi;
- penjelasan setiap bagian;
- keterbatasan;
- jawaban untuk pertanyaan yang mungkin muncul.

Istilah mengikuti tabel istilah di notebook 01.

---

## 1. Pesan utama

1. **Hasil.** 153 rekaman mentah diubah menjadi 195.469 *epoch* berlabel yang siap dipakai model. Jumlah per tahap dicocokkan dengan perhitungan terpisah di setiap langkah, jadi tidak ada label yang bergeser atau hilang.
2. **EEG dan EOG.** Kedua sinyal disaring ke rentang 0,3–35 Hz, lalu dinormalkan per rekaman memakai median dan rentang antar-kuartil. Normalisasi per rekaman diperlukan karena skala sinyal berbeda hingga 8 kali antar-rekaman.
3. **EMG.** Sinyal ini perlu cara khusus. 4,2% nilainya negatif karena masalah kalibrasi alat, dan skalanya berbeda hingga 73 kali antar-rekaman. Logaritma tidak bisa dipakai, sehingga dipilih *robust scaling* lalu `arcsinh`.
4. **Rekaman bermasalah.** 17 rekaman EMG-nya bermasalah, dan 11 di antaranya milik subjek berusia 80 tahun ke atas. Rekaman ini tidak dibuang, tetapi ditandai untuk dianalisis kemudian.

## 2. Posisi dalam proyek

Notebook 01 hanya membaca ringkasan. Notebook 02 adalah satu-satunya tempat sinyal mentah dibaca dan diubah. Hasilnya, satu berkas per rekaman di `data/processed/sc/`, dipakai oleh semua notebook sesudahnya.

Ringkasan dari notebook 01 berfungsi sebagai kunci jawaban. Label yang dibentuk di sini harus menghasilkan jumlah *epoch* yang sama persis.

```
Berkas mentah (EDF)  →  label per epoch  →  trimming  →  penyaringan  →  normalisasi  →  pemeriksaan mutu  →  simpan + verifikasi
     A–C                    D                 E             F             G–H                I                       J
```

## 3. Alur presentasi

| Urutan | Yang ditampilkan | Yang diucapkan | Angka kunci |
|---|---|---|---|
| 1 | Hipnogram dengan jendela *trimming* (bagian E) | Yang dibuang hanya bagian terjaga yang jauh dari tidur | 22,1 jam → 7,0 jam; *epoch* tidur tidak berubah |
| 2 | Contoh sinyal Wake/N3/REM (bagian D) | Setiap sinyal membawa informasi berbeda; EMG paling jelas memisahkan REM dari Wake | EMG REM sekitar 0,2 vs Wake sekitar 3,5 |
| 3 | Tabel perbandingan transformasi EMG (bagian G) | Alasan memilih *robust scaling* + `arcsinh` | Kemiringan −0,87; AUC 0,850 |
| 4 | Grafik mutu per rekaman (bagian I) | Skala antar-rekaman berbeda jauh sehingga normalisasi per rekaman diperlukan; EMG paling tidak stabil | EEG/EOG berbeda 4,5–7,9 kali; EMG 73,5 kali |
| 5 | Ringkasan verifikasi (bagian J) | Data tersimpan sama persis dengan hasil olahan | 195.469 *epoch*; galat penyimpanan maksimum 0,0078 |

## 4. Penjelasan per bagian

### A. Pengaturan

Semua angka tetap ditaruh di satu tempat:
- 30 detik per *epoch*;
- 100 Hz;
- jeda *trimming* 60 *epoch* (30 menit);
- label −1 untuk *epoch* yang diabaikan.

Pembaca berkas EDF diambil dari skrip eksplorasi (`scripts/edfx_eda.py`) yang sudah diuji pada 197 rekaman, jadi tidak ditulis ulang.

Label −1 dipakai untuk *Movement time* dan *epoch* yang tidak dinilai pakar. *Epoch* ini tetap ada di urutan, supaya urutan waktu yang dibaca Bi-LSTM tidak berlubang, tetapi tidak dihitung dalam pelatihan maupun penilaian.

### B. Memasangkan berkas sinyal dan berkas label

**Cara kerja.**
- Setiap berkas sinyal dipasangkan dengan berkas label yang tujuh karakter pertama namanya sama, misalnya `SC4001E`.
- Karakter kedelapan tidak dipakai karena berisi kode petugas yang berbeda-beda.
- Ada pemeriksaan bahwa setiap berkas sinyal punya tepat satu pasangan.

**Keluaran.** 153 pasangan dari 78 subjek. Hanya subjek 13, 36, dan 52 yang punya satu malam.

**Makna.** Kebanyakan subjek punya dua malam, sehingga data harus dibagi per subjek. Kalau dibagi per berkas, malam pertama seseorang bisa masuk data latih dan malam keduanya masuk data uji.

### C. Isi satu rekaman

| Kanal | Frekuensi | Penyaringan oleh alat |
|---|---|---|
| EEG Fpz-Cz, EEG Pz-Oz, EOG | 100 Hz | 0,5–100 Hz |
| EMG dagu | 1 Hz | Disaring di atas 16 Hz, disearahkan, lalu dihaluskan di bawah 0,7 Hz |

**Makna.**
- EMG di SC bukan sinyal mentah, melainkan selubung yang menggambarkan kekuatan otot per detik. Karena itu EMG punya cabang tersendiri di model dan cara normalisasi sendiri.
- Amplitudo EOG sekitar tiga kali EEG, jadi setiap kanal perlu dinormalkan terpisah.

### D. Satu label per *epoch*

**Cara kerja.** Berkas label menyimpan rentang waktu, misalnya "N2 dari detik 30.750 selama 390 detik". Rentang ini dipecah menjadi satu label per 30 detik dengan aturan berikut:
- **Tahap 3 dan 4** (standar lama Rechtschaffen & Kales) digabung menjadi N3 (standar AASM).
- ***Movement time* dan *epoch* yang tidak dinilai** diberi label −1.
- **Label dipotong mengikuti panjang sinyal.** Pada 150 dari 153 rekaman, label lebih panjang daripada sinyal; label tanpa sinyal tidak berguna.

**Keluaran pada seluruh rekaman.**
- 416.424 *epoch*: Wake 285.433, N1 21.522, N2 69.132, N3 13.039, REM 25.835.
- 1.463 *epoch* diabaikan.
- Semua angka cocok per rekaman dengan ringkasan notebook 01.

**Contoh sinyal (Wake, N3, REM).** Contoh diambil dari tengah rentetan terpanjang setiap tahap, supaya mewakili tahap yang stabil dan bukan *epoch* di perbatasan.

| Ciri | Terlihat pada | Makna |
|---|---|---|
| Gelombang lambat dan tinggi | EEG saat N3 | Gelombang delta, ciri tidur dalam |
| Gerakan cepat dan besar | EOG saat Wake | Gerakan mata dan kedipan |
| Gerakan mata | EOG saat REM | Gerakan mata cepat |
| EMG sangat rendah | EMG saat REM (sekitar 0,2 vs 3,5 saat Wake) | Otot lumpuh saat REM |

EEG saat REM mirip EEG saat Wake. Pembeda paling tegas di antara keduanya adalah EOG dan terutama EMG.

### E. *Trimming*

**Aturan.**
- Disisakan jendela dari 60 *epoch* sebelum *epoch* tidur pertama sampai 60 *epoch* sesudah *epoch* tidur terakhir. *Epoch* tidur adalah N1–REM.
- *Epoch* berlabel −1 di dalam jendela tetap disimpan.

**Keluaran.**

| Tahap | Sebelum | Sesudah | Porsi sebelum | Porsi sesudah |
|---|---|---|---|---|
| Wake | 285.433 | 65.941 | 68,8% | 33,7% |
| N1 | 21.522 | 21.522 | 5,2% | 11,0% |
| N2 | 69.132 | 69.132 | 16,7% | 35,4% |
| N3 | 13.039 | 13.039 | 3,1% | 6,7% |
| REM | 25.835 | 25.835 | 6,2% | 13,2% |

Jendela memuat 195.767 *epoch*. Sebanyak 298 di antaranya berlabel −1, sehingga tersisa 195.469 *epoch* berlabel.

**Dua lapis pemeriksaan.**
- Per rekaman, hasilnya cocok dengan perhitungan terpisah di skrip eksplorasi.
- Secara keseluruhan, jumlah N1–REM sama dengan angka di literatur.

Beberapa makalah melaporkan jumlah Wake yang sedikit berbeda, misalnya 69.824. Perbedaan ini berasal dari aturan *trimming* yang berbeda antar-penulis. Jumlah N1–REM yang identik menunjukkan bahwa perbedaannya hanya pada Wake.

### F. Penyaringan EEG dan EOG

**Alasan rentang 0,3–35 Hz.**
- Ciri tahap tidur berada di rentang ini, dari gelombang delta (0,5–4 Hz) sampai *spindle* (11–16 Hz) dan beta rendah.
- Di atas 35 Hz isinya kebanyakan derau otot dan listrik. Di bawah 0,3 Hz isinya pergeseran garis dasar.

**Cara kerja.**
- Filter Butterworth orde 4 dijalankan maju lalu mundur. Akibatnya gelombang tidak bergeser waktunya, sehingga pola penting seperti *K-complex* tetap berada di *epoch* yang benar.
- Penyaringan dilakukan pada jendela hasil *trimming*. Efek tepi filter hanya berlangsung beberapa detik dan jatuh di dalam 30 menit Wake di ujung jendela, sehingga tidak mengenai *epoch* tidur.

**Respons filter.**

| Frekuensi (Hz) | 0,1 | 0,3 | 1–10 | 30 | 35 | 40 | 45 |
|---|---|---|---|---|---|---|---|
| Penguatan (dB) | −76,6 | −6,0 | 0,0 | −0,5 | −6,0 | −31,7 | −81,5 |

**Makna.**
- −6 dB berarti amplitudo menjadi separuh.
- Karena alat perekam sudah menyaring di atas 0,5 Hz, efek nyata filter ini terutama memotong frekuensi di atas 35 Hz.
- Pada grafik spektrum, kurva sebelum dan sesudah penyaringan berhimpit pada 0,5–30 Hz.

EMG tidak disaring, karena informasinya ada pada tingkat aktivitas otot, bukan pada bentuk gelombang.

### G. Penanganan EMG

**Masalah yang terukur.**
- 4,2% nilai EMG negatif, tersebar di 119 rekaman. Selubung kekuatan otot secara fisik tidak mungkin negatif, jadi ini pergeseran kalibrasi alat.
- Median per rekaman berkisar dari −0,99 sampai 17,79.

**Perbandingan empat cara.**

| Cara | Nilai rusak | Kemiringan | AUC Wake vs REM |
|---|---|---|---|
| Mentah | 0,000% | 3,28 | 0,842 |
| Logaritma | 4,274% | −2,92 | 0,817 |
| *Robust scaling* | 0,357% | −2,42 | 0,838 |
| ***Robust scaling* + `arcsinh`** | **0,357%** | **−0,87** | **0,850** |

Keterangan kolom:
- **Kemiringan** (*skewness*): 0 berarti distribusi simetris.
- **AUC**: seberapa baik rerata EMG per *epoch* memisahkan Wake dari REM. 0,5 berarti setara menebak, 1 berarti sempurna.

**Keputusan.** *Robust scaling* per rekaman, nilai dibatasi ±20, lalu `arcsinh`.
- Fungsi `arcsinh` hampir linear di sekitar nol dan mirip logaritma untuk nilai besar, tetapi tetap terdefinisi untuk nilai negatif.
- Selisih AUC terhadap data mentah hanya 0,008. Jadi alasan utamanya adalah distribusi yang lebih simetris dan terbatas, yang memudahkan pelatihan jaringan saraf, bukan daya pisah yang lebih baik.

**Pemeriksaan.** Setelah diolah, EMG tetap turun berurutan dari Wake sampai REM pada seluruh rekaman.

### H. Pemotongan per *epoch* dan normalisasi

**Alasan *robust scaling*, bukan *z-score*.**
- *Z-score* memakai rerata dan simpangan baku, yang mudah rusak oleh satu artefak besar.
- *Robust scaling*, yaitu (x − median) / rentang antar-kuartil, tahan terhadap artefak.
- Normalisasi dihitung per rekaman dan per kanal, sesudah *trimming* dan penyaringan. Kalau dihitung sebelum *trimming*, statistiknya didominasi Wake di siang hari.

**Batas ±20.** Nilai di luar ±20 kali rentang antar-kuartil dipotong. Batas ini sangat longgar, setara sekitar ±27 simpangan baku pada data normal, sehingga hanya artefak ekstrem yang terkena.

**Bentuk data per rekaman.**

| Data | Bentuk | Keterangan |
|---|---|---|
| EEG | (*epoch*, 2, 3.000) | Fpz-Cz dan Pz-Oz. Keduanya disimpan agar kontribusi kanal kedua bisa diuji. |
| EOG | (*epoch*, 3.000) | |
| EMG | (*epoch*, 30) | Satu nilai per detik |
| Label, nomor *epoch* asli | (*epoch*,) | Nomor asli dipakai untuk percobaan kebocoran data |

**Pemeriksaan pada rekaman contoh.**
- Sesudah normalisasi, median EEG dan EOG 0 dan rentang antar-kuartilnya 1.
- Nilai terbesar EOG tepat 20, artinya ada artefak yang terpotong.

### I. Pemeriksaan mutu 153 rekaman

**Cara kerja.** Pengolahan dijalankan pada semua rekaman, tetapi hanya ukuran mutunya yang dicatat. Data baru ditulis ke disk sesudah semua pemeriksaan lolos.

Ada tiga tanda peringatan:
- **Pemotongan:** lebih dari 1% nilai satu kanal terpotong.
- **EEG datar:** EEG hampir konstan pada lebih dari 5% *epoch*, tanda elektroda lepas.
- **EMG bermasalah:** EMG nyaris datar, atau tidak lebih rendah saat REM dibanding Wake.

**Hasil.**
- EEG dan EOG bersih: tidak ada EEG datar, dan pemotongan paling banyak 0,35%.
- 20 rekaman ditandai: 9 karena pemotongan dan 17 karena EMG. Sebagian rekaman terkena keduanya, dan seluruh pemotongan yang berlebihan juga terjadi pada EMG.
- 11 dari 17 rekaman EMG bermasalah milik subjek 80 tahun ke atas, padahal kelompok ini hanya 21% rekaman.

**Keputusan.** Tidak ada rekaman yang dibuang. Membuangnya akan menghapus sebagian besar subjek tertua dan mengubah populasi. Tandanya disimpan, dan di notebook 05 terbukti model tidak lebih terganggu oleh rekaman ini dibandingkan *baseline*.

### J. Penyimpanan dan verifikasi

**Format.**
- Satu berkas `.npz` per rekaman, tanpa kompresi, karena sinyal yang berderau tidak terkompresi dengan baik.
- EEG dan EOG disimpan dalam presisi setengah (`float16`) agar ukurannya separuh. Karena nilainya dibatasi ±20, presisi ini cukup.

**Verifikasi.** Semua berkas dibaca ulang dan diperiksa:
- bentuk dan tipe data;
- tidak ada nilai rusak;
- label di rentang −1 sampai 4;
- nomor *epoch* berurutan;
- subjek dan malam sesuai;
- jumlah per tahap sama dengan bagian E.

**Hasil.**
- 153 berkas (3,55 GB) berisi 195.469 *epoch* berlabel.
- Galat terbesar akibat presisi setengah adalah 0,0078. Galat ini terjadi pada nilai artefak di sekitar 16–20; untuk nilai tipikal di sekitar 1, galatnya sekitar 0,0005.

Ringkasan per rekaman disimpan di `results/tables/02_cache_manifest.csv`. Berkas ini ikut di-*commit*, sehingga anggota tim bisa mencocokkan hasil olahan mereka tanpa saling mengirim data.

## 5. Keterbatasan yang perlu diakui

- **Normalisasi memakai statistik seluruh malam.** Ini sah untuk evaluasi karena tidak memakai label atau rekaman lain. Namun pada pemakaian waktu nyata, statistik itu baru tersedia setelah rekaman selesai.
- **Batas filter dan batas pemotongan tidak dioptimasi.** Keduanya dipilih dari konvensi dan alasan fisiologis.
- **Data ST belum diolah.** EMG di ST direkam berbeda, dan penyaringan alatnya juga berbeda. Keduanya perlu disamakan dulu sebelum uji eksternal.

## 6. Antisipasi pertanyaan

**"Kenapa normalisasi per rekaman, bukan per dataset?"**
Rentang antar-kuartil EEG berbeda hingga 8 kali antar-rekaman dan EMG hingga 73 kali. Perbedaan ini datang dari posisi elektroda, kondisi kulit, dan kalibrasi alat, bukan dari tahap tidur. Kalau dinormalkan per dataset, model bisa belajar mengenali rekamannya, bukan tahap tidurnya.

**"Apakah normalisasi per rekaman membocorkan informasi uji?"**
Tidak antar-subjek. Statistiknya hanya berasal dari rekaman itu sendiri dan tidak memakai label. Yang perlu diakui hanyalah bahwa statistik itu memakai seluruh malam (lihat keterbatasan).

**"Kenapa tidak membuang rekaman dengan EMG bermasalah?"**
Sebagian besar rekaman itu milik subjek tertua. Membuangnya akan mengubah populasi dan membuat hasil tidak sebanding dengan literatur. Analisis di notebook 05 menunjukkan keunggulan model justru lebih besar pada rekaman ini.

**"Kenapa filter dijalankan maju dan mundur?"**
Filter satu arah menggeser gelombang dalam waktu. Pergeseran beberapa ratus milidetik bisa memindahkan pola penting, misalnya *K-complex*, ke *epoch* tetangga.

**"Kenapa `float16`, apakah tidak merusak data?"**
Galat terbesarnya 0,0078, pada skala sinyal yang rentang antar-kuartilnya 1. Galat ini jauh di bawah derau alami sinyal. Sebagai gantinya, ukuran data menjadi separuh (3,55 GB). Ini penting karena data dibaca langsung dari disk selama pelatihan.

## 7. Lampiran: membaca kode

| Potongan kode | Fungsinya |
|---|---|
| `psg.name[:7]` | Mengambil kode rekaman untuk memasangkan berkas |
| `np.bincount(labels + 1)` | Menghitung jumlah per label. Label digeser satu karena `bincount` tidak menerima −1. |
| `labels[max(0, a - 60): min(n, b + 61)]` | Memotong jendela *trimming*. `+ 61` karena batas kanan irisan Python tidak ikut terpotong. |
| `signal.butter(4, BAND, btype="bandpass", output="sos")` | Merancang filter Butterworth orde 4 |
| `signal.sosfiltfilt` | Menjalankan filter maju lalu mundur |
| `np.clip((x - median) / iqr, -20, 20)` | *Robust scaling* dengan batas ±20 |
| `np.arcsinh(z)` | Pemampatan nilai EMG |
| `np.savez(path, eeg=..., labels=...)` | Menyimpan beberapa larik dalam satu berkas |

**Berkas.**
- Hasil olahan: `data/processed/sc/*.npz`.
- Ringkasan per rekaman: `results/tables/02_cache_manifest.csv`.

**Keputusan desain.** Dicatat di `docs/catatan/KEPUTUSAN_DESAIN.md`:
- K1: *epoch* tanpa label di dalam jendela;
- K2: dua kanal EEG;
- K4: penanganan EMG.

**Waktu jalan.** Sekitar 5,5 menit.
