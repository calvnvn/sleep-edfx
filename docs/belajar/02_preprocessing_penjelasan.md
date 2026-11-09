# Penjelasan Notebook 02_preprocessing, Bagian A sampai J: Dari Berkas EDF ke Data Siap Latih

Dokumen ini menemani `notebooks/02_preprocessing.ipynb`. Bagian A sampai D membentuk label per epoch. Bagian E sampai J memangkas jendela tidur (*trimming*), menyaring sinyal, menentukan penanganan EMG, menormalisasi, memeriksa mutu seluruh rekaman, dan menyimpan hasilnya sebagai *cache*.

## Bagian 1. Gambaran besar

### 1.1 Apa yang dikerjakan Fase 1

**Analogi.** Dapur restoran. Bahan mentah (berkas EDF, 7,1 GB) tidak boleh diubah. Tugas Fase 1 adalah mencuci, memotong, dan menakar bahan itu menjadi porsi siap masak, lalu menyimpannya di tempat terpisah (`data/processed/`) agar model dapat memakainya berulang kali.

**Teknis.** Satu rekaman diubah menjadi larik epoch 30 detik beserta label, dengan urutan berikut.

| Tahap | Bagian notebook | Status dokumen ini |
|---|---|---|
| Memasangkan berkas dan membaca struktur | A, B, C | Dibahas |
| Membentuk label per epoch | D | Dibahas |
| Memangkas jendela tidur (*trimming*) | E | Dibahas |
| Menyaring EEG dan EOG | F | Dibahas |
| Menangani EMG (keputusan K4) | G | Dibahas |
| Memotong epoch dan menormalisasi | H | Dibahas |
| Memeriksa mutu seluruh rekaman | I | Dibahas |
| Menyimpan dan memverifikasi | J | Dibahas |

### 1.2 Perbedaan dengan notebook EDA

Notebook 01 hanya **membaca ringkasan** (`eda_records.json`). Notebook 02 **membaca sinyal mentah** dan menghasilkan data yang diubah. Karena itu, di Bagian D kita memakai `eda_records.json` sebagai **kunci jawaban**: label yang kita bentuk sekarang harus menghasilkan hitungan yang sama dengan hasil EDA untuk seluruh 153 rekaman.

### 1.3 Aturan yang tidak boleh dilanggar

Berkas di `data/raw/` hanya dibaca, tidak pernah ditulis. Seluruh hasil disimpan di folder lain.

---

## Bagian 2. Penjelasan sel per sel

### Sel A. Persiapan dan konfigurasi

**Analogi.** Papan pengumuman di dapur: semua takaran tetap ditempel di satu tempat, sehingga koki tidak mengarang angka di tengah memasak.

**Isi.**

- `ROOT` dicari otomatis dengan menelusuri folder induk sampai menemukan `scripts/edfx_eda.py`, sehingga notebook berjalan dari folder mana pun.
- `from edfx_eda import read_header, read_signals, read_annotations` memakai ulang pembaca EDF yang sudah teruji pada 197 rekaman di tahap EDA. Kita tidak menulis ulang pembaca itu.
- Konstanta tetap: `EPOCH_S = 30`, `FS = 100`, `TRIM_EPOCHS = 60`, dan `IGNORE = -1`. Semua sel berikutnya membaca angka dari sini.
- Warna tiap tahap tidur sama dengan notebook 01, agar gambar kedua notebook terbaca sebagai satu kesatuan.

**Mengapa `IGNORE = -1`?** Ini wujud keputusan K1 (lihat `docs/catatan/KEPUTUSAN_DESAIN.md`): epoch *Unscored* dan *Movement time* tetap ada di urutan, tetapi diberi nilai −1 agar nanti tidak dihitung dalam *loss* maupun metrik.

### Sel B. Inventaris: memasangkan PSG dengan hypnogram

**Analogi.** Mencocokkan tiap kotak rekaman dengan catatan dokternya, memakai nomor di kotak.

**Cara kerja.**

1. `RAW.glob("*-PSG.edf")` mengumpulkan semua berkas sinyal.
2. Untuk tiap berkas, `psg.name[:7]` (misalnya `SC4001E`) dipakai mencari berkas hypnogram yang diawali 7 karakter yang sama. Karakter ke-8 tidak ikut karena berisi kode petugas yang bervariasi (`SC4001E0` dengan `SC4001EC`).
3. `assert len(hyp) == 1` memastikan tiap PSG memiliki **tepat satu** pasangan. Kode lama memakai `hyps[0]` tanpa memeriksa ini.
4. `subject` diambil dari karakter ke-4 dan 5, `night` dari karakter ke-6. Kode Python memakai indeks dari nol, sehingga `psg.name[3:5]` dan `psg.name[5]`.

**Pemeriksaan.** Ada 153 pasangan dan 78 subjek. Hanya subjek 13, 36, dan 52 yang memiliki satu malam; sisanya dua malam.

**Cara membaca keluaran.** Lima baris pertama menunjukkan `SC4001E` dan `SC4002E` milik subjek 0 (malam 1 dan 2). Ini alasan pembagian lipatan nanti harus berdasarkan **subjek**, bukan berkas: dua malam milik orang yang sama.

### Sel C1. Anatomi satu rekaman: header dan tata letak kanal

**Analogi.** Membaca label di kotak sebelum membukanya: isinya apa, direkam seberapa cepat, sudah disaring apa.

**Keluaran** untuk `SC4001E`: 2.650 *data record* × 30 detik = 22,08 jam, dan tabel tujuh kanal.

| Kanal | Satuan | Frekuensi | Prefilter (dari alat) |
|---|---|---|---|
| EEG Fpz-Cz | µV | 100 Hz | HP 0,5 Hz, LP 100 Hz |
| EEG Pz-Oz | µV | 100 Hz | sama |
| EOG horizontal | µV | 100 Hz | sama |
| EMG submental | µV | **1 Hz** | HP 16 Hz, rektifikasi, LP 0,7 Hz |

Kanal lain (pernapasan, suhu, penanda) tidak dipakai. Perhatikan baris EMG: nilainya sudah berupa envelope, bukan sinyal mentah. Itu sebabnya EMG diperlakukan berbeda dari EEG dan EOG di bagian berikutnya.

### Sel C2. Memuat empat sinyal

`read_signals` membaca data digital 16-bit lalu mengubahnya menjadi satuan fisik (µV) dengan skala linear dari header.

**Keluaran yang perlu diperhatikan** (`SC4001E`):

| Kanal | Sampel | Detik | Rentang |
|---|---|---|---|
| EEG Fpz-Cz | 7.950.000 | 79.500 | −192 sampai 171 µV |
| EOG horizontal | 7.950.000 | 79.500 | −512 sampai 517 µV |
| EMG submental | 79.500 | 79.500 | −0,1 sampai 3,8 |

Dua hal yang dapat dibaca dari tabel ini. Pertama, EEG 100 Hz memiliki 100 kali lebih banyak sampel daripada EMG 1 Hz, padahal durasinya sama. Kedua, amplitudo EOG sekitar tiga kali EEG. Ini salah satu alasan sinyal harus dinormalisasi per kanal.

### Sel D1. Dari rentang hypnogram ke label per epoch

**Analogi.** Dokter mencatat "tidur ringan dari menit 510 sampai 517", bukan satu catatan per 30 detik. Kita menguraikan rentang itu menjadi baris-baris absensi per 30 detik.

**Cara kerja `hypnogram_to_epochs`.**

1. `n_epochs = floor(durasi_sinyal / 30)`. Larik label **dipotong mengikuti panjang sinyal**, karena hypnogram dapat lebih panjang (150 dari 153 rekaman SC) dan label tanpa sinyal tidak berguna.
2. Larik diisi dahulu dengan `IGNORE`. Epoch yang tidak tercakup hypnogram otomatis diabaikan.
3. Untuk tiap anotasi: epoch awal `onset // 30`, epoch akhir `(onset + durasi) // 30`, lalu irisan itu diisi label hasil pemetaan.
4. Pemetaan `STAGE_TO_LABEL`: *Sleep stage 3* dan *4* digabung menjadi N3 (R&K ke AASM); *Movement time* dan *Sleep stage ?* menjadi `IGNORE`.

**Keluaran** `SC4001E`: 2.650 epoch (= ⌊79.500 ÷ 30⌋), dengan 1.997 Wake, 58 N1, 250 N2, 220 N3, 125 REM, dan 0 diabaikan. Hypnogram berakhir pada jam ke-24,0 sedangkan sinyal pada 22,1 jam; selisih itu terpotong.

**Satu trik yang sering membingungkan.** Pada hitungan, kita menulis `np.bincount(labels + 1)`. `np.bincount` hanya menerima bilangan tak negatif, jadi semua label digeser satu: −1 menjadi indeks 0 ("diabaikan"), 0 menjadi indeks 1 (Wake), dan seterusnya.

### Sel D2. Hypnogram satu malam

**Analogi.** Grafik detak jantung, tetapi untuk tahap tidur.

**Cara membaca gambar.**

- Sumbu y berurutan seperti hypnogram klinis: Wake di atas, N3 di bawah, REM di antara Wake dan N1.
- Garis berwarna adalah tahap tidur; garis abu-abu tipis menghubungkan perpindahan antar tahap.
- Area abu-abu di kanan menandai label yang melewati ujung sinyal dan dipotong.

**Temuan yang terlihat.** `SC4001E` terjaga selama 8,5 jam sebelum epoch tidur pertama dan tertidur sampai jam ke-14,5. N3 terkumpul di paruh pertama masa tidur (sekitar jam ke-9 sampai 13,7), sedangkan REM muncul berkala di sela siklus. Pola ini sesuai dengan tidur sehat, dan bagian terjaga sebelum jam ke-8,5 itulah yang akan dipangkas pada *trimming*.

### Sel D3. Sinyal pada tiga tahap

**Tujuan.** Memperlihatkan apa yang sesungguhnya akan dilihat model dalam satu epoch.

`middle_epoch_of_longest_run` mencari rentetan terpanjang suatu tahap, lalu mengambil epoch di tengahnya. Dengan begitu contoh yang dipilih adalah tahap yang stabil, bukan epoch di perbatasan.

**Cara membaca (kolom: Wake, N3, REM; baris: empat kanal).**

| Tanda | Terlihat pada | Makna |
|---|---|---|
| Gelombang lambat beramplitudo tinggi | EEG, kolom N3 | Aktivitas delta, ciri tidur dalam |
| Gerakan besar dan cepat | EOG, kolom Wake | Gerakan mata dan kedipan saat terjaga |
| Gelombang lambat beramplitudo besar | EOG, kolom REM | Gerakan mata |
| EMG sangat rendah | Baris EMG, kolom REM (sekitar 0,2 dibanding 3,5 saat Wake) | Kelumpuhan otot (*atonia*) khas REM |

Baris terakhir menjelaskan mengapa EMG dipertahankan sebagai cabang tersendiri meski hanya 30 sampel per epoch: ia pembeda REM yang paling tegas dari EEG yang mirip Wake. Pengaturan `sharey="row"` membuat tiap baris memakai skala yang sama antar kolom, sehingga perbedaan amplitudo antar tahap terlihat jujur.

### Sel D4. Label seluruh 153 rekaman dan uji silang dengan EDA

**Cara kerja.** Fungsi yang sama dijalankan pada semua rekaman, lalu dua `assert` memeriksa tiap rekaman terhadap `eda_records.json`:

1. Hitungan lima kelas (Wake sampai REM) sama dengan `full[:5]` dari EDA.
2. Jumlah epoch `IGNORE` sama dengan `full[5] + full[6]` (*Movement time* ditambah *Unscored*).

**Keluaran yang benar:**

| Kelas | Jumlah epoch |
|---|---|
| Diabaikan | 1.463 |
| Wake | 285.433 |
| N1 | 21.522 |
| N2 | 69.132 |
| N3 | 13.039 |
| REM | 25.835 |
| **Berlabel (lima kelas)** | **414.961** |
| **Seluruh epoch** | **416.424** |

Angka 414.961 adalah "SC-78 sebelum *trimming*" pada dokumen Ref-1, dan 1.463 sama dengan 1.335 (*Unscored*) + 128 (*Movement time*). Karena `assert` tidak gagal, pembentukan label kita terbukti setara dengan hasil EDA, sekaligus menegaskan bahwa belum ada epoch yang dipangkas pada tahap ini.

---

### Sel E1 sampai E3. *Trimming*: membuang Wake di luar jendela tidur

**Analogi.** Sebuah film dua jam yang diawali dan diakhiri lima belas menit layar kosong. Kita memotong layar kosong itu, tetapi menyisakan sedikit jeda di awal dan akhir agar transisi masuk dan keluar dari tidur tetap terlihat.

**Mengapa perlu.** Rekaman SC rata-rata berdurasi 22,7 jam karena perekam dibiarkan menyala sepanjang siang hari. Akibatnya Wake mencapai 68,8% dari seluruh epoch berlabel, dan sebagian besar Wake itu terjadi saat subjek beraktivitas biasa, bukan di sekitar tidur. Bagian ini tidak informatif bagi pembeda tahap tidur, mendominasi kelas, dan akan membiaskan statistik normalisasi bila tidak dibuang.

**Aturan (`sleep_window`).**

1. *Epoch tidur* adalah epoch berlabel 1 sampai 4 (N1, N2, N3, REM). Wake dan label −1 sengaja tidak dihitung sebagai tidur.
2. Jendela dimulai 60 epoch (30 menit) sebelum epoch tidur pertama dan berakhir 60 epoch sesudah epoch tidur terakhir.
3. `max(0, …)` dan `min(len(labels), …)` mencegah jendela melewati ujung rekaman. Angka `+ 1` pada batas atas diperlukan karena irisan Python bersifat eksklusif di ujung kanan.
4. Epoch berlabel −1 **di dalam** jendela tidak dibuang. Ini penerapan keputusan K1: urutan waktu harus tetap utuh bagi Bi-LSTM.

**Keluaran `SC4001E` (E1).** Tidur berlangsung dari epoch 1.021 sampai 1.741, sehingga jendelanya adalah [961, 1.802), yaitu 841 dari 2.650 epoch (7,0 jam dari 22,1 jam). Jumlah Wake turun dari 1.997 menjadi 188, sedangkan N1, N2, N3, dan REM tidak berubah sama sekali. Itu sifat penting *trimming*: ia hanya mengurangi Wake, tidak pernah menyentuh epoch tidur.

**Cara membaca gambar (E2).** Area abu-abu adalah bagian yang dibuang (Wake pagi dan siang), area biru muda adalah jendela yang dipertahankan. Garis hypnogram yang rapat berada seluruhnya di area biru, dan di luarnya hanya ada garis datar Wake.

**Uji unit (E3).** Ada dua lapis pemeriksaan, dan keduanya lolos.

- Per rekaman: hitungan lima kelas di dalam jendela sama dengan `trim` pada `eda_records.json`. Perhitungan EDA dikerjakan oleh fungsi `trim_wake` yang ditulis terpisah, sehingga kesamaan ini menjadi pengesahan silang.
- Agregat: 195.469 epoch berlabel dengan N1 21.522, N2 69.132, N3 13.039, dan REM 25.835 (sama dengan literatur). Jendela memuat 195.767 epoch, dan selisihnya, **298 epoch**, adalah epoch berlabel −1 yang tercatat pada keputusan K1.

| Kelas | Sebelum | Sesudah | Porsi sebelum | Porsi sesudah |
|---|---|---|---|---|
| Wake | 285.433 | 65.941 | 68,8% | 33,7% |
| N1 | 21.522 | 21.522 | 5,2% | 11,0% |
| N2 | 69.132 | 69.132 | 16,7% | 35,4% |
| N3 | 13.039 | 13.039 | 3,1% | 6,7% |
| REM | 25.835 | 25.835 | 6,2% | 13,2% |

Jumlah Wake 65.941 berbeda dari 69.824 yang dilaporkan sebagian makalah. Selisih itu bersumber dari aturan *trimming* yang berbeda antar penulis, bukan dari kekeliruan kita, dan jumlah N1 sampai REM yang identik membuktikannya. Karena itu aturan di atas ditulis eksplisit.

### Sel F1 dan F2. Penyaringan *band-pass* untuk EEG dan EOG

**Analogi.** *Equalizer* pada pemutar musik: frekuensi yang berisi informasi dibiarkan, dentum terlalu rendah dan desis terlalu tinggi diredam.

**Mengapa 0,3 sampai 35 Hz.** Ciri tahap tidur terletak pada gelombang delta (0,5 sampai 4 Hz), theta, alfa, *sleep spindle* (11 sampai 16 Hz), dan beta rendah. Di atas ±35 Hz energi sinyal didominasi derau otot dan listrik, sedangkan di bawah 0,3 Hz hanya ada pergeseran garis dasar (*drift*). Perangkat SC sudah memasang *high-pass* 0,5 Hz dan *low-pass* 100 Hz pada kanal EEG dan EOG, sehingga efek nyata filter kita adalah *low-pass* 35 Hz, sedangkan batas bawah 0,3 Hz hanya menjaga agar sisa *drift* tidak lolos.

**Cara kerja.**

- `signal.butter(4, BAND, btype="bandpass", fs=FS, output="sos")` merancang filter Butterworth orde 4 dalam bentuk *second-order sections*, yang stabil secara numerik. Butterworth dipilih karena responsnya datar di pita lolos.
- `signal.sosfiltfilt` menjalankan filter **maju lalu mundur**. Hasilnya berfase nol, yaitu gelombang tidak bergeser waktunya, sesuatu yang penting bagi struktur seperti *K-complex* yang harus tetap berada pada epoch yang benar. Konsekuensinya, gain dikuadratkan sehingga redaman dalam dB menjadi dua kali lipat.
- Sinyal diubah ke `float64` sebelum disaring demi presisi, lalu dikembalikan ke `float32`.
- Filter dijalankan pada jendela hasil *trimming*, bukan pada 24 jam penuh. Cara ini 3 sampai 4 kali lebih cepat, dan efek tepi hanya bertahan beberapa detik di dalam 30 menit Wake pada ujung jendela sehingga tidak menyentuh epoch tidur.

**Cara membaca tabel gain (F1).**

| Frekuensi (Hz) | 0,05 | 0,1 | 0,3 | 1 | 10 | 30 | 35 | 40 | 45 |
|---|---|---|---|---|---|---|---|---|---|
| Gain (dB) | −124,8 | −76,6 | −6,0 | 0,0 | 0,0 | −0,5 | −6,0 | −31,7 | −81,5 |

Antara 1 dan 10 Hz sinyal tidak berubah (0 dB). Pada tepi pita, 0,3 dan 35 Hz, amplitudo menjadi setengahnya (−6 dB), dan di luar tepi redaman naik sangat cepat. Frekuensi 50 Hz tidak dicantumkan karena tepat berada di batas Nyquist untuk laju 100 Hz.

**Cara membaca gambar (F2).** Sumbu keduanya logaritmik. Garis biru (hasil filter) dan abu-abu (mentah) berhimpit di rentang 0,5 sampai 30 Hz, artinya isi informasi tidak disentuh. Di kanan garis putus-putus 35 Hz garis biru jatuh tajam, sedangkan garis abu-abu mendatar. Datarnya garis abu-abu itu adalah lantai derau, yaitu bagian sinyal yang memang tidak membawa informasi tahap tidur. Sumbu y dipotong pada 10⁻³ agar pita lolos tetap terbaca, karena redaman filter jauh melampaui batas itu.

**EMG tidak disaring.** Alasannya sama dengan yang dicatat di Bagian C: nilai EMG di SC adalah *envelope* 1 Hz yang informasinya terletak pada level, sehingga *high-pass* justru akan membuangnya.

### Sel G1 sampai G3. Penanganan EMG (keputusan K4)

**Analogi.** Sepuluh dapur memakai timbangan dengan titik nol yang berbeda: sebagian menunjukkan −1 gram saat kosong, sebagian lagi 17 gram. Angka mentahnya tidak bisa dibandingkan, dan menerapkan "logaritma berat" pada timbangan yang menunjukkan angka negatif pasti gagal.

**Masalah yang terukur (G1).** Dari 153 rekaman:

- 4,2% nilai EMG di dalam jendela bernilai negatif, tersebar pada 119 rekaman. Sebuah *envelope* seharusnya tidak pernah negatif, sehingga ini berasal dari *offset* kalibrasi perangkat.
- Median per rekaman berkisar dari −0,99 sampai 17,79, dan IQR per rekaman dari 0,07 sampai 5,15. Skala antar-rekaman berbeda hingga puluhan kali.

Dua fakta itu menutup dua jalan: nilai mentah tidak dapat dipakai langsung (skala berbeda), dan logaritma langsung tidak dapat dipakai (ada nilai negatif).

**Empat kandidat (G2).** Kolom *values lost* adalah persentase nilai yang rusak atau terbuang oleh transformasi. *Skewness* mengukur kemiringan distribusi (0 berarti simetris). AUC Wake vs REM menunjukkan seberapa baik rerata EMG per epoch memisahkan kedua kelas itu (0,5 berarti tebakan, 1,0 berarti sempurna).

| Kandidat | Nilai terbuang | Skewness | AUC Wake vs REM |
|---|---|---|---|
| Mentah | 0,000% | 3,281 | 0,842 |
| Logaritma (batas bawah 10⁻³) | 4,274% | −2,921 | 0,817 |
| *Robust* (median dan IQR) | 0,357% | −2,421 | 0,838 |
| ***Robust* + arcsinh** | **0,357%** | **−0,874** | **0,850** |

**Cara membaca.**

- Logaritma membuang 4,3% nilai (semua yang ≤ 10⁻³, termasuk nilai negatif) dan menghasilkan AUC terendah. Kandidat ini gugur.
- *Robust scaling* saja mempertahankan informasi tetapi distribusinya masih miring (−2,4) karena rekaman dengan IQR sangat kecil menghasilkan nilai terstandardisasi yang ekstrem.
- Menambahkan `arcsinh` setelah *robust scaling* menurunkan kemiringan menjadi −0,87 dan memberi AUC tertinggi.

**Apa itu arcsinh.** Fungsi `arcsinh(z) = ln(z + √(z² + 1))` berperilaku hampir linear di sekitar nol dan hampir logaritmik untuk nilai besar, tetapi tetap terdefinisi untuk nilai negatif. Singkatnya, ia adalah "logaritma yang aman bagi bilangan negatif". Ia menekan ekor panjang tanpa membuang apa pun.

**Keputusan K4.** EMG di SC diproses dengan urutan: *robust scaling* per rekaman (median dan IQR), pemotongan ±20, lalu `arcsinh`. Perlu dicatat dengan jujur bahwa selisih AUC dengan data mentah hanya 0,008. Alasan utama memilih `arcsinh` adalah distribusi yang lebih simetris dan terbatas (sekitar ±3,7) bagi jaringan saraf, bukan lompatan kemampuan pemisah.

**Cara membaca gambar (G3).** Titik adalah median dan batang adalah rentang antar-kuartil EMG per tahap. Nilainya turun berurutan dari Wake ke REM. Itulah *atonia* REM yang dijelaskan pada D3, kini terbukti pada 153 rekaman dan bukan hanya satu contoh, sehingga cabang EMG memang layak dipertahankan.

### Sel H1 sampai H3. Pemotongan epoch dan normalisasi per rekaman

**Analogi.** Menakar porsi seragam sebelum memasak. Setiap bahan, dari rekaman mana pun, harus diukur dengan takaran yang sama agar model tidak belajar "siapa yang direkam" alih-alih "tahap apa".

**Mengapa *robust scaling*.** *Z-score* memakai rerata dan simpangan baku, dua besaran yang mudah terdistorsi satu artefak besar. *Robust scaling* memakai median dan IQR (persentil 75 dikurangi persentil 25), sehingga artefak tidak merusak skala. Rumusnya `(x − median) / IQR`, dihitung terpisah **per rekaman dan per kanal**.

**Aturan tambahan.**

- Statistik dihitung **setelah** *trimming* dan penyaringan. Bila dihitung pada 24 jam penuh, statistiknya didominasi Wake siang hari. Filter harus mendahului normalisasi karena filter mengubah amplitudo.
- Nilai di luar ±20 IQR dipotong (`CLIP = 20`). Rentang itu sangat longgar, setara sekitar ±27 simpangan baku pada data normal, sehingga hanya artefak ekstrem yang terkena.
- Pemotongan memakai `np.clip`, dan persentase nilai yang terpotong dicatat sebagai bahan pemeriksaan mutu.

**Fungsi `preprocess_record`.** Ini perangkaian seluruh langkah untuk satu rekaman: membaca sinyal, memotong ke jendela, menyaring EEG dan EOG, menormalisasi keempat kanal, menerapkan `arcsinh` pada EMG, lalu mengubah bentuk menjadi larik per epoch. Keluarannya:

| Larik | Bentuk | Keterangan |
|---|---|---|
| `eeg` | (N, 2, 3.000) | Fpz-Cz dan Pz-Oz (dua kanal disimpan agar K2 dapat diuji sebagai ablasi) |
| `eog` | (N, 3.000) | EOG horizontal |
| `emg` | (N, 30) | Satu nilai per detik, tanpa *upsampling* |
| `labels` | (N,) | 0 sampai 4, dan −1 untuk epoch yang diabaikan |
| `epoch_idx` | (N,) | Nomor epoch asli dalam rekaman, dibutuhkan keputusan K3 |

**Keluaran `SC4001E` (H1).** Bentuknya (841, 2, 3.000), (841, 3.000), dan (841, 30). IQR mentah EEG Fpz-Cz adalah 19,67 µV, Pz-Oz 12,61 µV, dan EOG 17,95 µV, tiga angka yang berbeda cukup jauh, dan itulah yang diseragamkan.

**Pemeriksaan (H2).** Setelah normalisasi, median EEG dan EOG tepat 0 dan IQR tepat 1,000. IQR EMG 0,961 karena `arcsinh` sedikit memampatkan nilai, dan itu wajar. Nilai maksimum EEG hanya 8,4 dan 12,4, sedangkan EOG mencapai 20,0, tepat di batas pemotongan, yang berarti ada satu atau beberapa artefak yang terpotong.

**Cara membaca gambar (H3).** Gambar ini sejajar dengan D3 tetapi memakai data yang sudah diolah. Pola yang sama tampak: gelombang lambat amplitudo besar pada N3, gerak mata pada EOG saat Wake dan REM, serta EMG paling rendah pada REM. Setiap baris berbagi skala sumbu y sendiri, karena skala EMG sengaja berbeda dari EEG dan EOG setelah `arcsinh`.

### Sel I1 sampai I3. Pemeriksaan mutu seluruh 153 rekaman

**Analogi.** Menimbang dan memeriksa setiap porsi sebelum dimasukkan ke kulkas. Satu bahan rusak yang lolos ke penyimpanan akan merusak setiap masakan yang memakainya.

**Cara kerja.**

- I1 menjalankan `preprocess_record` pada seluruh rekaman (±130 detik) dan hanya mencatat statistik per rekaman, tanpa menyimpan larik. Dengan begitu pemeriksaan selesai sebelum 3,55 GB ditulis ke disk.
- I2 menerapkan tiga `assert` (semua nilai terhingga, jumlah epoch berlabel dan per kelas sama dengan hasil E3, panjang jendela sama dengan hasil E3) dan tiga tanda peringatan:
  - `flag_clip`: lebih dari 1% sampel pada satu kanal terpotong.
  - `flag_flat`: EEG hampir konstan pada lebih dari 5% epoch (indikasi elektroda lepas).
  - `flag_emg`: IQR mentah EMG di bawah 0,1 (nyaris datar), atau rerata EMG REM tidak lebih rendah daripada Wake.

**Hasil.** Seluruh nilai terhingga dan seluruh hitungan cocok. EEG dan EOG bersih: tidak ada rekaman dengan EEG datar, dan persentase pemotongan terbesar hanya 0,35% (Pz-Oz) serta 0,17% (EOG). Sebanyak 20 rekaman ditandai, **semuanya karena EMG**: 9 karena pemotongan dan 17 karena kualitas EMG (ada yang tumpang tindih). Temuan yang menarik, 11 dari 17 rekaman EMG bermasalah berasal dari subjek berusia 80 tahun ke atas, padahal kelompok itu hanya 32 dari 153 rekaman (±21%). Pada sebagian lansia, sinyal EMG submental memang lemah atau berubah.

**Keputusan.** Tidak ada rekaman yang dibuang. Membuang subjek lansia mengubah populasi dan membuat hasil tidak sebanding dengan literatur. Sebagai gantinya, tanda-tanda itu disimpan di berkas manifes agar tahap ablasi dapat menguji sensitivitas, misalnya menghitung hasil dengan dan tanpa 17 rekaman tersebut.

**Cara membaca gambar (I3).** Panel kiri memperlihatkan IQR mentah tiap rekaman. EEG dan EOG berbeda hingga 4,5 sampai 7,9 kali antar rekaman, sedangkan EMG hingga 73,5 kali. Itu pembenaran empiris untuk normalisasi per rekaman. Panel kanan memperlihatkan persentase sampel terpotong: hampir semua titik EEG dan EOG berada di nol, dan hanya EMG yang memiliki beberapa titik tinggi (maksimum 17,75% pada `SC4432E`).

### Sel J1 dan J2. Penyimpanan *cache* dan verifikasi muat ulang

**Analogi.** Porsi yang sudah ditakar dimasukkan ke kotak berlabel di kulkas. Memasak berikutnya tidak perlu mengulang mencuci dan memotong.

**Keputusan format.**

- **Satu berkas `.npz` per rekaman** di `data/processed/sc/`. Pembagian lipatan nanti berdasarkan subjek, sehingga memuat per rekaman memudahkan merakit lipatan tanpa memuat semuanya.
- **Tanpa kompresi.** Sinyal biologis yang berderau tidak terkompresi baik, sedangkan penulisan dan pembacaan menjadi jauh lebih lambat.
- **`float16` untuk EEG dan EOG**, `float32` untuk EMG. `float16` memangkas ukuran separuhnya. Nilai kita dibatasi ±20 sehingga presisinya memadai: galat maksimum terukur 0,0078 (terjadi pada nilai di sekitar 16 sampai 20, yaitu artefak), sedangkan nilai tipikal di sekitar 1 berselisih sekitar 0,0005.

**Isi tiap berkas.**

| Kunci | Tipe | Keterangan |
|---|---|---|
| `eeg`, `eog`, `emg` | float16, float16, float32 | Seperti pada tabel H |
| `labels`, `epoch_idx` | int8, int32 | Label per epoch dan nomor epoch asli |
| `subject`, `night`, `age` | int16, int8, int16 | Dibutuhkan untuk lipatan *subject-wise* (K3) dan stratifikasi usia |
| `window` | int32 (2) | Awal dan akhir jendela *trimming* pada rekaman asli |
| `norm` | float32 (4, 2) | Median dan IQR mentah untuk Fpz-Cz, Pz-Oz, EOG, EMG |

**Hasil.** 153 berkas, total **3,55 GB**. Waktu penulisan ±157 detik.

**Verifikasi (J2).** Seluruh berkas dibaca ulang (±13 detik) lalu diperiksa:

1. Jumlah berkas 153 dan seluruh bentuk serta tipe data sesuai tabel di atas.
2. Tidak ada nilai NaN atau tak hingga.
3. Label berada pada −1 sampai 4, dan nomor epoch berurutan tanpa celah.
4. Subjek dan malam di tiap berkas sama dengan inventaris.
5. Jumlah per kelas seluruh berkas: Wake 65.941, N1 21.522, N2 69.132, N3 13.039, REM 25.835, total **195.469**, sama persis dengan hasil E3.
6. Perbandingan terhadap perhitungan baru untuk `SC4001E`: galat `float16` di bawah 0,01.

Berkas `results/tables/02_cache_manifest.csv` menyimpan ringkasan per rekaman (jumlah epoch, jumlah per kelas, statistik mutu, tanda peringatan, ukuran berkas). Berkas ini kecil dan ikut di-*commit*, sehingga teman satu kelompok dapat membandingkan hasil *cache* mereka dengan milik pengembang tanpa mengirim data.

**Waktu total.** Satu kali jalan Bagian A sampai J membutuhkan sekitar 5,5 menit pada mesin pengembang (I1 ±130 detik dan J1 ±157 detik adalah bagian terlama).

---

## Bagian 3. Istilah yang muncul

| Istilah | Arti dalam notebook ini |
|---|---|
| Epoch | Potongan 30 detik; satuan label dan satuan masukan model |
| Hypnogram | Berkas anotasi tahap tidur dari dokter, berupa rentang waktu |
| `pathlib.Path` | Cara modern memegang alamat berkas; operator `/` menyambung folder |
| `glob` | Mencari berkas dengan pola nama |
| `assert` | Pernyataan yang harus benar; bila salah, notebook berhenti dengan pesan galat |
| `np.diff` | Selisih elemen bersebelahan; dipakai mencari titik tempat label berubah |
| `np.bincount` | Menghitung kemunculan tiap bilangan bulat 0, 1, 2, ... |
| `sharey="row"` | Subplot satu baris berbagi skala sumbu y |
| Jendela tidur | Rentang epoch dari 30 menit sebelum tidur pertama sampai 30 menit sesudah tidur terakhir |
| Butterworth | Rancangan filter dengan respons datar di pita lolos |
| Fase nol (`sosfiltfilt`) | Filter maju lalu mundur sehingga gelombang tidak bergeser waktunya |
| Welch (PSD) | Cara memperkirakan sebaran daya sinyal menurut frekuensi |
| IQR | Selisih persentil 75 dan persentil 25; ukuran sebaran yang tahan terhadap nilai ekstrem |
| *Robust scaling* | Pengurangan median lalu pembagian IQR |
| *Clipping* | Memotong nilai di luar batas (di sini ±20) pada batas itu |
| `arcsinh` | Fungsi mirip logaritma yang tetap terdefinisi untuk nilai negatif |
| *Skewness* | Ukuran kemiringan distribusi; 0 berarti simetris |
| AUC | Kemampuan memisahkan dua kelas; 0,5 tebakan, 1,0 sempurna |
| `float16` | Bilangan pecahan 16 bit; separuh ukuran `float32` dengan presisi sekitar tiga digit |
| `.npz` | Berkas NumPy yang menampung beberapa larik bernama |
| Manifes | Tabel ringkasan per berkas hasil *cache* untuk pemeriksaan silang |

## Bagian 4. Percobaan kecil

1. Ganti `REC = "SC4001E"` dengan `"SC4762E"`, lalu jalankan ulang D1 sampai D3. Rekaman ini memiliki rentetan hampir 50 menit tanpa label; lihat bagaimana area abu-abu muncul di hypnogram.
2. Ubah `picks` di D3 menjadi `{"N1": 1, "N2": 2}` untuk melihat mengapa N1 sulit dibedakan dari Wake secara visual.
3. Hapus `+ 1` pada `np.bincount(labels + 1, ...)` dan baca pesan galatnya.
4. Ubah `TRIM_EPOCHS` menjadi 20, jalankan ulang E3, dan baca `assert` mana yang gagal serta berapa jumlah Wake yang baru.
5. Ubah batas atas `BAND` dari 35 menjadi 20 Hz, jalankan F2, dan lihat bagian spektrum mana yang ikut terpotong.
6. Pada G2, tambahkan kandidat *z-score* biasa (`(x - x.mean()) / x.std()`) dan bandingkan skewness serta AUC-nya.
7. Ganti `REC` menjadi `"SC4611E"` (EMG hampir datar) lalu jalankan H1 sampai H3 dan perhatikan baris EMG.

## Bagian 5. Pertanyaan untuk menguji pemahaman

1. Mengapa pasangan berkas dicocokkan dengan 7 karakter pertama, bukan seluruh nama?
2. Mengapa larik label dipotong mengikuti panjang sinyal, bukan sebaliknya?
3. Apa akibatnya bila epoch tak berlabel dibuang dari larik, bukan diberi −1?
4. Mengapa angka 414.961 di D4 belum mencerminkan *trimming*?
5. Pada D3, mengapa EMG saat REM jauh lebih rendah, dan mengapa itu berguna bagi model?
6. Mengapa N1, N2, N3, dan REM tidak berubah jumlahnya setelah *trimming*, sedangkan Wake turun drastis?
7. Mengapa filter dijalankan maju dan mundur, dan apa akibatnya bila hanya maju?
8. Mengapa logaritma langsung tidak dapat dipakai pada EMG, dan apa yang diberikan `arcsinh` sebagai gantinya?
9. Mengapa statistik normalisasi dihitung setelah *trimming* dan penyaringan, bukan sebelumnya?
10. Mengapa 20 rekaman yang ditandai pada I2 tidak dibuang, dan untuk apa tandanya disimpan?
11. Mengapa `epoch_idx`, `subject`, dan `night` ikut disimpan di tiap berkas *cache*?
