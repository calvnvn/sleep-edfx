# Pembaruan Progres: Tahap *Pre-processing* (Bagian E sampai J) dan Rencana Berikutnya

**Proyek:** *Automated Sleep Stage Classification using Multimodal Hybrid Deep Learning*
**Mata kuliah:** *Deep Learning* (AI706), Informatika, Universitas Multimedia Nusantara, Kelompok 8
**Disusun:** 9 November 2026 (diperbarui dengan penjelasan konsep tiap tahap)

Dokumen ini ditulis agar dapat dibaca sendiri tanpa dokumen lain. Bagian 1 sampai 3 memberi ringkasan, konteks, dan posisi dalam kerangka penelitian. Bagian 4 menjelaskan dasar teknis yang dikerjakan sebelumnya (Bagian A sampai D pada notebook `02_preprocessing`). Bagian 5 menjelaskan secara rinci pekerjaan minggu ini (Bagian E sampai J), termasuk alur data dan cara membaca setiap keluaran. Bagian 6 mencatat revisi pada dokumen kerangka penelitian, Bagian 7 menguraikan rencana minggu depan, Bagian 8 mencatat keterbatasan, Bagian 9 mengantisipasi pertanyaan, dan Bagian 10 berisi skrip singkat untuk dibacakan kepada dosen. Seluruh angka berasal dari keluaran notebook yang telah dijalankan pada seluruh 153 rekaman, bukan dari literatur, kecuali ditandai dengan sitasi.

---

## 1. Ringkasan

Minggu ini *pipeline pre-processing* untuk seluruh rekaman *Sleep-Cassette* (SC) selesai dibangun, diuji, dan dijalankan. Hasilnya adalah satu berkas data siap latih untuk setiap rekaman (153 berkas, total 3,55 GB) yang memuat 195.469 *epoch* berlabel, sama persis dengan angka acuan pada tahap analisis data eksplorasi (EDA) dan literatur. Terdapat empat hasil utama:

- **Pemangkasan jendela tidur (*trimming*):** *Epoch Wake* turun dari 285.433 menjadi 65.941, sehingga porsi *Wake* turun dari 68,8% menjadi 33,7%, sedangkan jumlah N1, N2, N3, dan REM tidak berubah.
- **Keputusan penanganan EMG (K4):** Transformasi logaritma pada EMG tidak dapat dipakai karena 4,2% nilai EMG negatif. Pilihan akhirnya adalah *robust scaling* per rekaman, pemotongan nilai ekstrem pada ±20, lalu transformasi `arcsinh`.
- **Pemeriksaan mutu:** Kanal EEG dan EOG bersih pada seluruh rekaman, sedangkan 17 rekaman ditandai karena kualitas EMG yang rendah (11 di antaranya berasal dari subjek berusia 80 tahun ke atas).
- **Verifikasi:** Setiap berkas dibaca ulang dan diperiksa; jumlah *epoch* per kelas sama dengan hasil *trimming*.

Belum ada hasil model apa pun (baik *deep learning* maupun *baseline*). Tahap berikutnya adalah ekstraksi fitur dan pelatihan model (Bagian 7).

---

## 2. Konteks Proyek

### 2.1 Tujuan dan data

Proyek ini mengklasifikasikan lima tahap tidur menurut standar AASM, yaitu *Wake*, N1, N2, N3, dan REM, dari sinyal polisomnografi. Satuan data adalah *epoch* berdurasi 30 detik. Model yang diusulkan adalah *Multi-Branch 1D-CNN* (cabang EEG, EOG, dan EMG) yang digabung dengan *Bi-LSTM* dua lapis dan dilatih dengan *Focal Loss*, dievaluasi memakai validasi berbasis subjek (*subject-wise GroupKFold*) dengan *Macro-F1* dan *Cohen's Kappa* sebagai metrik utama.

Dataset yang dipakai adalah PhysioNet *Sleep-EDF Expanded* v1.0.0 (Kemp dkk., 2000), yang terdiri atas 197 rekaman dalam dua kelompok:

- **Sleep-Cassette (SC):** 153 rekaman dari 78 subjek sehat berusia 25 sampai 101 tahun, direkam di rumah memakai perekam kaset termodifikasi. Kelompok ini menjadi data utama pelatihan dan evaluasi (disebut SC-78). Subset 20 subjek pertama (39 rekaman, usia 25 sampai 34 tahun) disebut SC-20 dan dipakai untuk iterasi cepat.
- **Sleep-Telemetry (ST):** 44 rekaman dari 22 subjek dengan kesulitan tidur ringan dalam studi efek *Temazepam*. Kelompok ini hanya dipakai sebagai uji generalisasi di luar distribusi pelatihan (*out-of-distribution*) dan tidak dibahas pada dokumen ini.

### 2.2 Tiga sumber informasi dalam satu rekaman

Setiap rekaman terdiri atas dua berkas EDF, dan isi berkas sinyal sendiri terbagi dua bagian. Pembedaan ini penting untuk memahami dari mana setiap angka berasal.

- **Header berkas sinyal (`*-PSG.edf`):** Metadata tentang rekaman, yaitu daftar kanal, laju sampel, satuan, durasi, faktor konversi ke satuan fisik, dan catatan *prefilter*. Header tidak memuat nilai sinyal.
- **Data sinyal (di berkas yang sama):** Nilai sebenarnya dari kanal EEG, EOG, dan EMG. Pembacaannya memakai faktor konversi dari header.
- ***Hypnogram* (`*-Hypnogram.edf`):** Anotasi dokter berupa rentang waktu per tahap tidur, bukan satu label per *epoch*.

### 2.3 Kanal dan sifat EMG

Kanal yang dipakai adalah EEG Fpz-Cz, EEG Pz-Oz, dan EOG horizontal (semuanya 100 Hz, 3.000 sampel per *epoch*), serta EMG submental. EMG pada SC bukan sinyal mentah. Header mencatat *prefilter* `HP:16Hz Rectification LP:0.7Hz`, yang artinya alat perekam sendiri telah melakukan tiga tahap sebelum mendigitalkan sinyal:

1. ***High-pass* 16 Hz:** meloloskan komponen yang lebih cepat dari 16 Hz sehingga gerakan lambat dan *drift* hilang dan yang tersisa adalah getaran aktivitas otot.
2. ***Rectification* (penyearahan):** nilai negatif dibalik menjadi positif, serupa nilai mutlak, sehingga ayunan naik-turun menjadi nilai positif.
3. ***Low-pass* 0,7 Hz:** menghaluskan hasilnya menjadi rata-rata kekuatan per beberapa detik.

Hasilnya adalah ***amplitude envelope***: satu nilai per detik (1 Hz, atau 30 sampel per *epoch*) yang menyatakan kekuatan otot dagu. Sinyal mentahnya tidak dapat dipulihkan, sehingga EMG diperlakukan berbeda dari EEG dan EOG. Catatan tambahan pada *header* EEG dan EOG berbunyi `[enhanced cassette BW]`; keterangan ini tidak dijelaskan pada dokumentasi dataset yang diperiksa. *BW* kemungkinan berarti *bandwidth* (lebar pita frekuensi perekam kaset), tetapi hal ini bersifat tafsiran dan tidak memengaruhi analisis, karena yang dipakai adalah `HP:0.5Hz LP:100Hz`.

### 2.4 Label −1 (*ignore label*)

Keputusan K1 menetapkan bahwa *epoch* berlabel *Sleep stage ?* (*Unscored*) dan *Movement time* tidak dibuang dari urutan, melainkan diberi nilai −1 sehingga dikecualikan dari perhitungan *loss* dan metrik. Analogi yang membantu adalah kursi bioskop yang rusak: kursinya tetap dipasang agar nomor kursi lain tidak bergeser, tetapi tidak dihitung sebagai penonton.

- **Mengapa tidak dibuang:** *Bi-LSTM* membaca urutan *epoch* berkelanjutan. Bila sebuah *epoch* di tengah malam dibuang, *epoch* sebelum dan sesudahnya tampak bersebelahan padahal terpisah, sehingga konteks waktunya keliru.
- **Mengapa tidak dijadikan kelas keenam:** Model akan belajar memprediksi kelas tambahan dan metrik ikut tercampur. Nilai −1 berada di luar rentang kelas 0 sampai 4, sehingga mudah disaring dengan syarat `label ≠ −1`.
- **Sumber label −1:** *Movement time*, *Sleep stage ?*, dan *epoch* yang tidak tercakup *hypnogram*. Label *hypnogram* yang melewati ujung sinyal tidak diberi −1 melainkan dipotong, karena tidak ada sinyal untuk *epoch* itu.
- **Angka:** Seluruh rekaman memuat 1.463 *epoch* berlabel −1 (1.335 *Unscored* ditambah 128 *Movement time*); di dalam jendela *trimming* tersisa 298.
- **Dampak pada pelatihan:** *Loss* (termasuk *Focal Loss*) dan semua metrik dihitung hanya pada *epoch* berlabel 0 sampai 4.

---

## 3. Posisi dalam Kerangka Penelitian

Dokumen kerangka penelitian terdiri atas tujuh poin. Tabel berikut memetakan setiap poin ke tahap pengerjaan dan statusnya.

| Poin kerangka penelitian | Tahap pengerjaan atau bagian notebook | Status |
|---|---|---|
| 1. *Exploratory Data Analysis* | Notebook `01_eda` | Selesai (sebelum minggu ini) |
| 2. *Pre-processing Pipeline*, butir 1 (*30-Minute Wake Trimming*) | Notebook 02, **Bagian E** | **Selesai minggu ini** |
| 2. *Pre-processing Pipeline*, butir 2 (*AASM Label Mapping*) | Notebook 02, Bagian B sampai D | Selesai (minggu lalu) |
| 2. *Pre-processing Pipeline*, butir 3 (penanganan EMG) | Notebook 02, **Bagian G** | **Selesai minggu ini** |
| 2. *Pre-processing Pipeline*, butir 4 (*denoising* dan *scaling*) | Notebook 02, **Bagian F dan H** | **Selesai minggu ini** |
| Pemeriksaan mutu dan penyimpanan *cache* (belum ada pada kerangka) | Notebook 02, **Bagian I dan J** | **Selesai minggu ini** |
| 3. *Feature Engineering Strategy* | Pembangunan *Epoch-level Feature Extractor* (tiga cabang 1D-CNN) | Rencana minggu depan |
| 4. *Proposed Model Architecture* | Penggabungan fitur dan *Temporal Sequence Modeler* (*Bi-LSTM*) | Rencana minggu depan (sebagian) |
| 5. *Evaluation Matrix and Validation Scheme* | Pelatihan dengan *loss* terpilih, validasi berbasis subjek, dan perhitungan metrik | Tahap berikutnya |
| 6. *End-to-End Pipeline* | Seluruh tahap | Berjalan |
| 7. *Research Gap* | Eksperimen ablasi terkontrol dan uji generalisasi pada data ST | Tahap berikutnya |

Dengan demikian seluruh poin *Pre-processing Pipeline* pada kerangka penelitian kini telah diimplementasikan dan diuji pada kode. Bagian I dan J merupakan tambahan rekayasa yang belum tercantum pada kerangka dan perlu ditambahkan (lihat Bagian 6).

---

## 4. Dasar Teknis dari Minggu Lalu (Bagian A sampai D)

Minggu lalu pekerjaan berfokus pada pemahaman data dan pembentukan label yang benar. Bagian ini merangkum tahap-tahap tersebut karena Bagian E sampai J bergantung padanya.

### 4.1 EDA

Dataset diverifikasi langsung dari berkas EDF. Dihitung rata-rata waktu tiap tahap tidur dan dibuat visualisasi kondisi data (distribusi kelas sebelum dan sesudah *trimming*, durasi rekaman, komposisi tahap menurut usia). Ditemukan bahwa EMG pada SC berupa *envelope* 1 Hz, bukan sinyal mentah, sehingga tidak boleh di-*upsample* maupun disaring *band-pass*.

### 4.2 Bagian A dan B: konfigurasi dan pemasangan berkas

**Konfigurasi.** Sel pertama memuat pustaka, menemukan folder repositori, mengimpor pembaca EDF dari `scripts/edfx_eda.py`, dan menetapkan parameter tetap (30 detik per *epoch*, 100 Hz untuk EEG dan EOG, 60 *epoch* untuk *trimming*, label −1 untuk *epoch* yang diabaikan). Sel-sel berikutnya membaca parameter dari sini sehingga angka tidak tersebar.

**Pemasangan berkas.** Berkas sinyal dicari dengan pola nama (`*-PSG.edf`), yang menghasilkan 153 alamat berkas. Untuk tiap sinyal, tujuh karakter pertama nama berkas (misalnya `SC4001E`) dipakai mencari berkas *hypnogram* yang diawali karakter yang sama. Karakter kedelapan tidak diikutkan karena berisi kode yang berbeda antar berkas. Sel ini hanya menyusun daftar alamat dan belum membaca isi berkas.

Sebuah `assert` memastikan setiap sinyal memiliki **tepat satu** pasangan. Nol pasangan berarti catatan label hilang, sedangkan dua pasangan atau lebih berarti pemasangan ambigu dan label dapat tertukar tanpa disadari. Pada data ini tidak ada yang menyimpang: 153 pasangan dari 78 subjek, dan hanya subjek 13, 36, dan 52 yang memiliki satu malam. Karena satu subjek dapat memiliki dua rekaman, pembagian lipatan nanti harus berdasarkan subjek, bukan berkas.

### 4.3 Bagian C: anatomi satu rekaman

Satu rekaman contoh (SC4001E) dibaca sebagai wakil seluruh SC. Perwakilan ini sah karena hasil EDA telah memeriksa susunan kanal seluruh rekaman dan seluruh 153 rekaman SC memiliki susunan yang identik (EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal pada 100 Hz, dan EMG pada 1 Hz). Pada tahap ini *header* dibaca (2.650 *data record* × 30 detik = 22,08 jam, tujuh kanal), kemudian data sinyal empat kanal dimuat. Dua pengamatan penting: EEG dan EOG memiliki 7.950.000 sampel sedangkan EMG hanya 79.500 untuk durasi yang sama, dan amplitudo EOG (sekitar −512 sampai 517 µV) kira-kira tiga kali EEG, sehingga setiap kanal perlu dinormalisasi.

### 4.4 Bagian D: label per *epoch*

**D1: dari rentang waktu ke label per *epoch*.** *Hypnogram* mencatat "tahap X dari detik A sampai B", bukan satu label per 30 detik. Fungsi pembentuk label membuat larik sepanjang jumlah *epoch* sinyal (`floor(durasi / 30)`), mengisinya dengan −1, lalu mengisi irisan *epoch* tiap anotasi dengan label hasil pemetaan (*Stage* 3 dan 4 digabung menjadi N3 menurut AASM; *Movement time* dan *Sleep stage ?* menjadi −1). Contoh tiga anotasi pertama SC4001E:

| Anotasi (onset, durasi, tahap) | *Epoch* yang diisi |
|---|---|
| 0 detik, 30.630 detik, *Wake* | 0 sampai 1.020 (1.021 *epoch*) → label 0 |
| 30.630 detik, 120 detik, tahap 1 | 1.021 sampai 1.024 (4 *epoch*) → label 1 |
| 30.750 detik, 390 detik, tahap 2 | 1.025 sampai 1.037 (13 *epoch*) → label 2 |

Hasil rekaman ini: 2.650 *epoch* dengan *Wake* 1.997, N1 58, N2 250, N3 220, REM 125. *Hypnogram* berakhir pada jam ke-24,0 sedangkan sinyal pada jam ke-22,1, sehingga selisih 1,9 jam dipotong karena label tanpa sinyal tidak berguna.

**D2: *hypnogram*.** Sumbu y disusun menurut konvensi klinis: *Wake* di atas, lalu REM, N1, N2, dan N3 di bawah. Susunan ini bukan hukum alam melainkan kebiasaan penyajian: sumbu dibaca sebagai "kedalaman tidur" dengan *Wake* sebagai keadaan sadar dan N3 sebagai tidur terdalam. REM ditaruh di antara *Wake* dan N1 karena EEG saat REM mirip sinyal terjaga atau tidur ringan, otak sangat aktif, dan seseorang lebih mudah terbangun daripada saat N3. Pada SC4001E, subjek terjaga sekitar 8,5 jam sebelum *epoch* tidur pertama dan tertidur sampai jam ke-14,5; N3 terkumpul di paruh awal tidur dan REM muncul berkala, pola khas tidur sehat.

**D3: sinyal tiga tahap.** Dipilih satu *epoch* di tengah bentangan terpanjang untuk *Wake*, N3, dan REM, lalu keempat kanalnya digambar. Ketiga tahap dipilih karena paling kontras: *Wake* sebagai kondisi dasar, N3 sebagai yang terdalam, dan REM sebagai tahap yang paling sulit dibedakan dari *Wake* lewat EEG sehingga memerlukan EOG dan EMG. Sinyal yang digambar adalah data sinyal mentah (bukan *header*). Cara membaca bentuk gelombang:

| Tanda | Terlihat pada | Makna |
|---|---|---|
| Gelombang lambat beramplitudo tinggi | EEG saat N3 | Aktivitas delta (0,5 sampai 4 Hz); neuron berayun serempak, ciri tidur dalam |
| EEG rendah dan rapat | EEG saat *Wake* dan REM | Neuron aktif sendiri-sendiri; sebab itu sulit dibedakan hanya dari EEG |
| Lonjakan besar dan cepat | EOG saat *Wake* | Kedipan dan lirikan mata |
| Gerakan mata khas | EOG saat REM | *Rapid eye movement*, asal nama REM |
| Nilai sangat rendah | EMG saat REM | *Atonia* REM: otot rangka dilumpuhkan otak |
| Nilai tinggi | EMG saat *Wake* | Otot dagu aktif |

**D4: uji silang.** Fungsi pembentuk label dijalankan pada 153 rekaman dan dua `assert` membandingkannya dengan hasil EDA (hitungan lima kelas dan jumlah *epoch* berlabel −1). Hasilnya 416.424 *epoch*: *Wake* 285.433, N1 21.522, N2 69.132, N3 13.039, REM 25.835, dan 1.463 diabaikan. Angka 414.961 *epoch* berlabel adalah SC-78 sebelum *trimming*.

---

## 5. Pembaruan Minggu Ini: Bagian E sampai J

Pekerjaan minggu ini merangkai pemangkasan jendela tidur, penyaringan, penanganan EMG, normalisasi, pemeriksaan mutu, dan penyimpanan menjadi satu fungsi per rekaman (`preprocess_record`), lalu menjalankannya pada seluruh 153 rekaman. Urutan tahap penting. Penyaringan dilakukan sebelum normalisasi karena filter mengubah amplitudo, dan normalisasi dilakukan setelah *trimming* agar statistiknya tidak didominasi data siang hari.

### 5.0 Alur data: kapan tiap hal dibaca, dipotong, dan disimpan

Bagian E hanya menentukan jendela dan memotong **label** di memori. Sinyal baru dipotong pada fungsi `preprocess_record` (Bagian H), dan berkas data baru tertulis pada Bagian J.

| Bagian | Membaca berkas dari disk? | Yang dikerjakan | Hasil disimpan ke `data/processed`? |
|---|---|---|---|
| E1 sampai E3 | Tidak | Menentukan jendela tiap rekaman (`windows`) dan memotong label (`trimmed`) dari label yang sudah ada di memori | Tidak |
| F | Tidak (memakai sinyal contoh di memori) | Merancang filter dan mengujinya pada satu rekaman | Tidak |
| G1 | Ya (EMG 153 rekaman) | Memeriksa dan membandingkan transformasi EMG | Tidak |
| H1 sampai H3 | Ya (satu rekaman) | Menjalankan `preprocess_record` pada SC4001E; **pemotongan sinyal pertama** | Tidak (hanya di memori) |
| I1 sampai I3 | Ya (153 rekaman) | Menjalankan fungsi yang sama untuk mencatat statistik mutu; datanya dibuang | Tidak |
| **J1** | Ya (153 rekaman) | Menjalankan fungsi yang sama dan **menulis hasilnya** | **Ya**, 153 berkas `.npz` |
| J2 | Ya (membaca ulang berkas) | Memverifikasi isi *cache* dan menyimpan manifes | Hanya manifes kecil di `results/tables/` |

Berkas di `data/raw/` tidak pernah diubah pada tahap mana pun. Fungsi yang sama dijalankan tiga kali (H, I, J) secara sengaja: H memastikan fungsi benar pada satu rekaman, I memastikan seluruh rekaman sehat sebelum ada yang ditulis, dan J menyimpan. Dengan urutan ini tidak ada berkas rusak yang sempat tertulis ke disk. Satu kali jalan seluruh notebook memerlukan sekitar 5,5 sampai 6 menit.

### 5.1 Bagian E: *Trimming* jendela tidur

Tahap ini bertujuan membuang bagian rekaman yang terjaga di luar waktu tidur. Rekaman SC rata-rata berdurasi 22,7 jam karena alat dibiarkan menyala sepanjang hari, sehingga *Wake* mencapai 68,8% dari seluruh *epoch* berlabel. Sebagian besar *Wake* itu terjadi saat subjek beraktivitas biasa, tidak informatif untuk membedakan tahap tidur, dan membuat kelas sangat tidak seimbang.

**Aturan.** *Epoch* tidur didefinisikan sebagai *epoch* berlabel N1, N2, N3, atau REM. *Wake* dan label −1 tidak dihitung sebagai tidur. Jendela yang dipertahankan dimulai 60 *epoch* (30 menit) sebelum *epoch* tidur pertama dan berakhir 60 *epoch* sesudah *epoch* tidur terakhir. Batas jendela dijaga agar tidak melewati ujung rekaman. *Epoch* berlabel −1 di dalam jendela tidak dibuang, sesuai keputusan K1.

**Sel E1 dan E2 (satu rekaman contoh).** Keduanya tidak membaca berkas apa pun; mereka memakai larik label SC4001E yang sudah ada di memori dari Bagian D. *Epoch* tidur pertama berada pada nomor 1.021 dan yang terakhir pada nomor 1.741, sehingga jendelanya mencakup *epoch* 961 sampai 1.801 (841 *epoch*, atau 7,0 jam dari 22,1 jam). Jumlah *Wake* turun dari 1.997 menjadi 188, sedangkan N1, N2, N3, dan REM tidak berubah. Ini sifat penting *trimming*: ia hanya mengurangi *Wake* dan tidak pernah menyentuh *epoch* tidur. Gambar pada E2 menampilkan *hypnogram* dengan area biru muda untuk jendela yang dipertahankan dan area abu-abu untuk yang dibuang.

**Sel E3 (seluruh 153 rekaman).** Sel ini juga tidak membaca berkas; ia memakai larik label seluruh rekaman (`all_labels`) dari Bagian D. Untuk tiap rekaman dihitung jendelanya (`windows`, hanya dua angka: awal dan akhir), lalu label dipotong ke jendela itu (`trimmed`). **Pada tahap ini hanya label yang dipotong; sinyal belum disentuh.** Pemotongan sinyal terjadi pada fungsi `preprocess_record` di Bagian H dengan memakai jendela ini. Itu sebabnya E3 harus lolos uji lebih dulu: semua tahap sesudahnya bergantung pada jendela ini.

**Hasil pada seluruh 153 rekaman.** Jendela seluruh rekaman memuat 195.767 *epoch*, terdiri atas 195.469 *epoch* berlabel dan 298 *epoch* berlabel −1.

| Kelas | Sebelum *trimming* | Sesudah *trimming* | Porsi sebelum | Porsi sesudah |
|---|---|---|---|---|
| *Wake* | 285.433 | 65.941 | 68,8% | 33,7% |
| N1 | 21.522 | 21.522 | 5,2% | 11,0% |
| N2 | 69.132 | 69.132 | 16,7% | 35,4% |
| N3 | 13.039 | 13.039 | 3,1% | 6,7% |
| REM | 25.835 | 25.835 | 6,2% | 13,2% |

**Verifikasi.** Dua lapis uji dijalankan dan keduanya lolos. Pertama, hitungan lima kelas pada setiap rekaman dibandingkan dengan hasil EDA yang dihitung oleh kode terpisah (`scripts/edfx_eda.py`). Kecocokan ini membuktikan dua implementasi yang ditulis terpisah menghasilkan jawaban yang sama, meskipun keduanya memakai definisi yang sama (tidur = N1 sampai REM, ±60 *epoch*). Kedua, hitungan agregat dibandingkan dengan angka acuan literatur (N1 = 21.522, N2 = 69.132, N3 = 13.039, REM = 25.835) dan dengan angka 298 *epoch* berlabel −1 pada keputusan K1. Jumlah *Wake* (65.941) berbeda dari 69.824 yang dilaporkan sebagian publikasi. Perbedaan itu bersumber dari aturan *trimming* yang berbeda antar penulis, dan jumlah N1 sampai REM yang identik menguatkannya. Karena itu aturan di atas ditulis secara eksplisit agar hasil dapat direproduksi.

### 5.2 Bagian F: Penyaringan *band-pass* untuk EEG dan EOG

Tahap ini bertujuan menyisakan rentang frekuensi yang membawa informasi tahap tidur. Analogi yang membantu adalah *equalizer* musik: nada yang membawa informasi dibiarkan, dentum terlalu rendah dan desis terlalu tinggi diredam. **Frekuensi** adalah seberapa cepat gelombang naik-turun (satuan Hz). Ciri tahap tidur terletak pada gelombang delta (0,5 sampai 4 Hz), theta, alfa, *sleep spindle* (sekitar 11 sampai 16 Hz), dan beta rendah. Di atas sekitar 35 Hz energi sinyal didominasi derau otot dan listrik, sedangkan di bawah 0,3 Hz hanya terdapat pergeseran garis dasar (*drift*) akibat keringat atau elektroda yang bergeser.

**Letak di kode.** Pada sel F1, baris `signal.butter(4, BAND, btype="bandpass", fs=FS, output="sos")` merancang filter Butterworth orde 4 dengan pita `BAND = (0,3; 35,0)` Hz pada laju sampel 100 Hz. Fungsi `bandpass(x)` yang memanggil `signal.sosfiltfilt` kemudian menerapkannya pada sinyal. Fungsi ini dipanggil pada sel F2 (untuk gambar spektrum) dan pada `preprocess_record` di Bagian H untuk tiga kanal: EEG Fpz-Cz, EEG Pz-Oz, dan EOG. Tidak ada pemanggilan untuk EMG.

**Mekanisme.** Filter dijalankan **maju dan mundur** (*zero-phase*). Penyaringan dua arah menghasilkan fase nol sehingga gelombang tidak bergeser waktunya, hal yang penting bagi struktur seperti *K-complex* yang harus tetap berada pada *epoch* yang benar. Konsekuensinya, gain filter dikuadratkan sehingga redaman dalam desibel menjadi dua kali lipat. Filter dirancang dalam bentuk *second-order sections* dan dijalankan dalam presisi `float64` demi stabilitas numerik. Filter diterapkan pada jendela hasil *trimming*, bukan pada 24 jam penuh, sehingga komputasinya sekitar tiga sampai empat kali lebih cepat. Transien filter pada ujung jendela diperkirakan hanya bertahan beberapa detik di dalam 30 menit *Wake* sehingga tidak menyentuh *epoch* tidur.

**Karakteristik terukur (sel F1).**

| Frekuensi (Hz) | 0,05 | 0,1 | 0,3 | 1 | 10 | 30 | 35 | 40 | 45 |
|---|---|---|---|---|---|---|---|---|---|
| Gain (dB) | −124,8 | −76,6 | −6,0 | 0,0 | 0,0 | −0,5 | −6,0 | −31,7 | −81,5 |

Gain 0 dB berarti sinyal tidak berubah, −6 dB berarti amplitudo tinggal setengah, dan −31,7 dB berarti tinggal sekitar 3%. Pada rentang 1 sampai 10 Hz sinyal tidak berubah, pada tepi pita (0,3 dan 35 Hz) amplitudo menjadi setengahnya, dan di luar tepi redaman naik sangat cepat.

**Cara membaca gambar spektrum (sel F2).** Gambar membandingkan spektrum daya (*power spectral density*, PSD) EEG Fpz-Cz dan EOG sebelum dan sesudah disaring pada jendela SC4001E. PSD menyatakan seberapa kuat sinyal pada tiap frekuensi, serupa tinggi batang pada *equalizer*. Kedua sumbu berskala logaritmik, sehingga tiap garis berarti kelipatan 10. Garis abu-abu adalah sinyal mentah, garis biru adalah sinyal sesudah disaring, dan dua garis putus-putus menandai batas filter (0,3 dan 35 Hz). Tiga wilayah dapat dibaca:

- **Di antara kedua garis putus-putus (sekitar 0,5 sampai 30 Hz):** garis biru berhimpit dengan abu-abu, artinya isi sinyal tidak diubah. Pada EEG tampak tonjolan kecil sekitar 12 sampai 13 Hz yang kemungkinan *sleep spindle* dan tetap utuh.
- **Di kiri garis 0,3 Hz:** garis biru turun tajam, artinya *drift* lambat dibuang.
- **Di kanan garis 35 Hz:** garis abu-abu mendatar (sekitar 0,15 pada EEG dan sekitar 1 pada EOG), sedangkan garis biru terjun tajam. Bagian datar itu bukan informasi, melainkan **lantai derau** (*noise floor*), yaitu getaran acak yang selalu ada pada alat ukur.

Sumbu y dipotong pada 10⁻³ agar pita lolos tetap terbaca, karena redaman filter jauh melampaui batas itu. Filter mulai meredam sedikit sebelum garis 0,3 Hz, bukan tepat pada garisnya, karena redamannya bertahap; perangkat perekam juga sudah memasang *high-pass* 0,5 Hz sehingga kurva mentah sendiri sudah mulai turun di sana.

**Mengapa EMG tidak disaring.** Terdapat tiga alasan:

1. **EMG pada SC sudah berupa *envelope*:** alat perekam telah menyaring dan menghaluskannya (Bagian 2.3), sehingga tidak ada getaran cepat yang perlu dibersihkan lagi.
2. **Informasinya ada pada tingkat (*level*) sinyal:** pembeda REM dari *Wake* adalah otot tegang (nilai tinggi) versus lumpuh (nilai rendah), yaitu komponen yang sangat lambat. Filter *high-pass* membuang komponen lambat, termasuk tingkat rata-ratanya. Analogi: menyaring grafik volume musik akan membuang tingkat rata-rata dan menyisakan naik-turun kecilnya, padahal perbedaan keras dan pelan itulah yang dicari. Buktinya, rerata EMG per tahap turun berurutan dari *Wake* ke REM (Bagian 5.3), yang merupakan perbedaan tingkat antar-*epoch*.
3. **Batas atas 35 Hz tidak bermakna:** EMG hanya 1 sampel per detik sehingga frekuensi tertinggi yang dapat diwakili (batas Nyquist) adalah 0,5 Hz, dan filter dengan batas 35 Hz bahkan tidak dapat dirancang untuk laju sampel itu.

Sebagai pengganti penyaringan, EMG tetap diolah dengan cara yang tidak menghapus tingkat sinyalnya (Bagian 5.3).

### 5.3 Bagian G: Penanganan EMG (keputusan K4)

Tahap ini bertujuan memilih cara yang tepat untuk mengolah EMG sebelum dimasukkan ke cabang EMG. Analogi yang membantu: sepuluh dapur memakai timbangan dengan titik nol berbeda, sebagian menunjukkan −1 gram saat kosong dan sebagian 17 gram. Angka mentahnya tidak sebanding, dan "logaritma berat" gagal pada timbangan yang menunjukkan angka negatif. Rancangan awal pada kerangka penelitian memakai transformasi `log(clip(x, ε))`. Pemeriksaan pada seluruh 153 rekaman menunjukkan rancangan itu bermasalah.

**Masalah yang terukur.**

- **Nilai negatif:** 4,2% seluruh sampel EMG di dalam jendela *trimming* bernilai negatif, tersebar pada 119 dari 153 rekaman. Sebuah *envelope* secara teori tidak pernah negatif. Transformasi logaritma tidak terdefinisi untuk nilai tersebut.
- **Skala antar-rekaman:** Median EMG per rekaman berkisar dari −0,99 sampai 17,79, dan IQR dari 0,07 sampai 5,15, sehingga nilai antar-rekaman tidak sebanding tanpa normalisasi.

**Dugaan penyebab nilai negatif (belum terbukti).** Dokumentasi dataset tidak menjelaskan sumbernya. Dugaan yang cocok dengan pola data adalah *offset* (titik nol yang bergeser) pada rangkaian perekam, bukan sinyal otot sungguhan. Contohnya, rekaman SC4611E seluruh nilainya negatif (berkisar −1,22 sampai 0,13) dan nyaris datar (IQR 0,07), sedangkan rekaman lain bermedian 3 sampai 6 namun tetap memuat nilai negatif kecil, yang konsisten dengan *offset* ditambah derau. Sumber pastinya (rangkaian analog kaset atau konversi ke digital) tidak dapat dipastikan. Kesimpulan yang aman hanyalah bahwa nilai EMG pada SC tidak dapat dibaca sebagai kekuatan otot absolut dan skalanya berbeda antar-rekaman.

**Perbandingan empat kandidat.** Setiap kandidat diterapkan pada EMG seluruh rekaman. Tiga ukuran dihitung dari rerata EMG per *epoch*: persentase nilai yang rusak atau terbuang oleh transformasi, kemiringan distribusi (*skewness*, 0 berarti simetris), dan AUC pemisah *Wake* dari REM (0,5 berarti tebakan, 1,0 sempurna), dihitung dari gabungan seluruh *epoch* berlabel.

| Kandidat | Nilai terbuang | *Skewness* | AUC *Wake* vs REM |
|---|---|---|---|
| Data mentah | 0,000% | 3,281 | 0,842 |
| Logaritma (batas bawah 10⁻³) | 4,274% | −2,921 | 0,817 |
| *Robust scaling* | 0,357% | −2,421 | 0,838 |
| ***Robust scaling* + `arcsinh`** | **0,357%** | **−0,874** | **0,850** |

**Keputusan K4.** EMG diproses dengan urutan: *robust scaling* per rekaman (dikurangi median lalu dibagi IQR), pemotongan nilai pada ±20, kemudian transformasi `arcsinh`. Fungsi `arcsinh(z) = ln(z + √(z² + 1))` berperilaku hampir linear di sekitar nol dan hampir logaritmik untuk nilai besar, tetapi tetap terdefinisi untuk nilai negatif (Burbidge dkk., 1988). Dengan demikian ia menekan ekor distribusi yang panjang tanpa membuang nilai apa pun.

**Alasan.** Logaritma gugur karena membuang 4,274% nilai dan menghasilkan AUC terendah. *Robust scaling* saja mempertahankan informasi, tetapi distribusinya masih miring (−2,421) akibat rekaman ber-IQR sangat kecil yang menghasilkan nilai terstandardisasi ekstrem. Penambahan `arcsinh` menurunkan kemiringan menjadi −0,874 dan memberi AUC tertinggi.

**Catatan jujur.** Selisih AUC terhadap data mentah hanya 0,008. Alasan utama memilih `arcsinh` adalah distribusi yang lebih simetris dan terbatas (sekitar ±3,7) bagi jaringan saraf, bukan peningkatan daya pisah yang besar.

**Validasi fisiologis.** Median rerata EMG per tahap turun berurutan dari *Wake* ke REM (setelah transformasi: *Wake* sekitar 0,13, N1 sekitar 0,03, N2 sekitar −0,15, N3 sekitar −0,2, REM sekitar −0,5). Pola ini mencerminkan *atonia* REM, yaitu kelumpuhan otot rangka saat REM, sehingga cabang EMG layak dipertahankan.

### 5.4 Bagian H: Pemotongan *epoch* dan normalisasi per rekaman

Tahap ini bertujuan menyeragamkan skala setiap kanal agar model belajar membedakan tahap tidur, bukan membedakan perangkat atau pemasangan elektroda. Amplitudo mentah sangat bervariasi: pada SC4001E, IQR mentah EEG Fpz-Cz adalah 19,67 µV, EEG Pz-Oz 12,61 µV, dan EOG 17,95 µV. Analogi: lini produksi dapur yang menerima bahan mentah lalu mengeluarkan porsi seragam siap masak. Bagian H membangun lini produksinya (fungsi `preprocess_record`) dan mengujinya pada **satu rekaman** (SC4001E), sedangkan penjalanan massal dilakukan pada Bagian I dan J.

**Alur kerja `preprocess_record`.** Fungsi ini menerima satu rekaman dan menjalankan langkah berikut secara berurutan:

1. **Membaca sinyal** keempat kanal dari berkas EDF.
2. **Memotong sinyal ke jendela** (`windows` dari Bagian E). Inilah pemotongan sinyal yang pertama.
3. **Menyaring EEG dan EOG** dengan `bandpass` (Bagian F). EMG dilewati.
4. **Normalisasi per kanal** dengan *robust scaling*: dikurangi median dan dibagi IQR, lalu dipotong pada ±20. Berbeda dari *z-score* yang memakai rerata dan simpangan baku, median dan IQR tidak mudah terdistorsi oleh satu artefak besar. Rentang ±20 IQR sangat longgar (setara sekitar ±27 simpangan baku pada data normal) sehingga hanya artefak ekstrem yang terkena.
5. **Transformasi `arcsinh` untuk EMG** (keputusan K4).
6. **Menyusun larik per *epoch*** dengan label dan nomor *epoch* asli dari `all_labels` sesuai jendela.

Urutan ini tidak boleh ditukar: filter harus mendahului normalisasi karena filter mengubah amplitudo, dan normalisasi harus dilakukan setelah pemotongan agar statistiknya tidak didominasi *Wake* siang hari. Karena statistik dihitung per rekaman dari rekaman itu sendiri, tidak ada informasi yang bocor antar-subjek maupun antar-*fold*.

**Keluaran per rekaman.**

| Larik | Bentuk | Keterangan |
|---|---|---|
| `eeg` | (N, 2, 3.000) | Fpz-Cz dan Pz-Oz; dua kanal disimpan agar pilihan satu atau dua kanal dapat diuji sebagai ablasi (K2 ditunda ke pemodelan) |
| `eog` | (N, 3.000) | EOG horizontal |
| `emg` | (N, 30) | Satu nilai per detik, tanpa *upsampling* |
| `labels` | (N) | 0 sampai 4, dan −1 untuk *epoch* yang diabaikan |
| `epoch_idx` | (N) | Nomor *epoch* asli dalam rekaman |

**Sel H1 (membangun dan mencoba fungsi).** Fungsi dijalankan pada SC4001E dengan hasil bentuk (841, 2, 3.000), (841, 3.000), dan (841, 30), serta statistik mentah (IQR EEG Fpz-Cz 19,67 µV, Pz-Oz 12,61 µV, EOG 17,95 µV, EMG 2,02; belum ada nilai terpotong). Kesimpulannya, satu fungsi yang sama nanti dipakai untuk seluruh rekaman, dan IQR mentah ketiga kanal yang berbeda cukup jauh itulah yang diseragamkan.

**Sel H2 (memeriksa hasil).** Setelah normalisasi, median EEG dan EOG tepat 0 dan IQR tepat 1,000, sedangkan IQR EMG 0,961 karena `arcsinh` sedikit memampatkan nilai. Nilai absolut maksimum EEG hanya 8,4 dan 12,4, sedangkan EOG mencapai 20,0 (tepat di batas pemotongan) yang menunjukkan adanya artefak yang terpotong. Kesimpulannya, normalisasi bekerja sesuai rancangan.

**Sel H3 (melihat sinyal hasil olahan).** Gambar menampilkan *epoch* *Wake* (epoch 991), N3 (epoch 1.145), dan REM (epoch 1.656) dari data yang sudah diolah. Pola yang tampak sama dengan Bagian D: N3 bergelombang lambat beramplitudo tinggi pada kedua kanal EEG, EOG berlonjak saat *Wake* dan menunjukkan ayunan khas gerakan mata saat REM, dan EMG paling rendah saat REM. Setiap baris memakai skala sumbu y sendiri karena skala EMG sengaja berbeda setelah `arcsinh`.

**Perbedaan dengan gambar Bagian D.** Gambar D3 menampilkan sinyal **mentah** (satuan µV) dari seluruh rekaman, sedangkan H3 menampilkan sinyal **terolah** (tanpa satuan). Epoch yang dipilih juga tidak sepenuhnya sama, karena pemilihannya memakai bentangan terpanjang di cakupan yang berbeda (seluruh rekaman pada D3, jendela *trimming* pada H3):

| Tahap | D3 | H3 | Sama? |
|---|---|---|---|
| N3 | epoch 1.145 | epoch 1.145 | Ya |
| REM | epoch 1.656 | epoch 1.656 | Ya |
| *Wake* | epoch 510 | epoch 991 | **Tidak** |

*Wake* terpanjang di seluruh rekaman adalah 1.021 *epoch* terjaga sebelum tidur (titik tengahnya epoch 510), sedangkan bagian itu sudah dibuang oleh *trimming*, sehingga H3 hanya dapat memilih *Wake* dari 60 *epoch* terakhir sebelum tidur (epoch 991). Judul gambar H3 pada notebook ("the same epochs") karena itu kurang tepat untuk *Wake* dan lebih tepat dibaca sebagai "tahap yang sama".

### 5.5 Bagian I: Pemeriksaan mutu seluruh 153 rekaman

Tahap ini bertujuan menemukan rekaman bermasalah sebelum data ditulis ke disk. Analogi: pabrik makanan menimbang dan memeriksa setiap kotak sebelum dimasukkan ke gudang, karena satu kotak rusak yang lolos akan memengaruhi setiap eksperimen berikutnya. Pemeriksaan ini sengaja dipisah dari penulisan (Bagian J) karena murah (sekitar 2 sampai 3 menit dan tanpa menulis apa pun), sehingga masalah ketahuan sebelum 3,55 GB dihasilkan.

**Sel I1 (mengumpulkan "rapor" tiap rekaman).** Fungsi `preprocess_record` dijalankan pada seluruh rekaman. Datanya dibuang, dan yang disimpan hanya satu baris rapor per rekaman pada tabel `qc`: jumlah *epoch* dan jumlah per kelas, IQR mentah tiap kanal, persentase nilai yang terpotong per kanal, persentase *epoch* EEG yang hampir datar, selisih rerata EMG *Wake* dikurangi REM, serta apakah seluruh nilai terhingga.

**Sel I2 (uji dan tanda peringatan).** Dua jenis pemeriksaan dibedakan. Uji (`assert`) menjaga kebenaran proses dan menghentikan notebook bila gagal: seluruh nilai terhingga, jumlah *epoch* berlabel dan per kelas sama dengan Bagian E, dan panjang jendela tiap rekaman sama dengan Bagian E. Tanda peringatan (*flag*) menjaga kualitas data dan tidak menghentikan notebook:

- **Pemotongan berlebih:** lebih dari 1% sampel pada satu kanal terpotong.
- **EEG datar:** EEG hampir konstan pada lebih dari 5% *epoch* (indikasi elektroda lepas).
- **EMG bermasalah:** IQR mentah EMG di bawah 0,1 (nyaris datar), atau rerata EMG saat REM tidak lebih rendah daripada saat *Wake*.

Ambang-ambang tersebut adalah ketetapan penulis sebagai penyaring awal, bukan standar baku, sehingga jumlah rekaman yang ditandai bergantung padanya.

**Hasil.** Seluruh nilai terhingga dan seluruh hitungan cocok. Kanal EEG dan EOG bersih: tidak ada rekaman dengan EEG datar dan persentase pemotongan terbesar hanya 0,35% (EEG Pz-Oz) serta 0,17% (EOG). Sebanyak 20 rekaman ditandai, **seluruhnya karena EMG**: 9 rekaman karena pemotongan EMG dan 17 rekaman karena kualitas EMG (ada yang tumpang tindih). Dari 17 rekaman EMG bermasalah, 6 memiliki IQR di bawah 0,1 dan 13 memiliki rerata REM yang tidak lebih rendah dari *Wake*.

**Sel I3 dan cara membaca gambar.** Gambar terdiri atas dua panel; setiap titik adalah satu rekaman (disebar acak ke kiri-kanan agar tidak bertumpuk, sehingga hanya tinggi titik yang bermakna).

- **Panel kiri** menampilkan IQR mentah per rekaman pada skala logaritmik. Awan titik yang sempit berarti skala antar-rekaman mirip; awan yang lebar berarti skalanya berbeda jauh. Selisih terbesar antar-rekaman adalah 4,5 kali (EEG Fpz-Cz), 6,8 kali (EOG), 7,9 kali (EEG Pz-Oz), dan **73,5 kali (EMG)**, dengan IQR EMG berkisar 0,07 sampai 5,15.
- **Panel kanan** menampilkan persentase sampel yang terpotong. Hampir semua titik EEG dan EOG berada sekitar 0%, sedangkan pada EMG ada beberapa rekaman yang melonjak: satu rekaman sekitar 17,75% (SC4432E) dan beberapa lainnya sekitar 3 sampai 8%.

Pesan gambar ini ada dua. Pertama, skala sinyal antar-rekaman berbeda jauh, sehingga setiap rekaman harus dinormalisasi sendiri; jika semua rekaman dibagi satu angka yang sama, rekaman ber-IQR kecil akan tampak datar dan yang ber-IQR besar tampak meledak padahal bedanya hanya pada skala perekam. Kedua, EMG adalah kanal yang paling rapuh. Grafik hanya memperlihatkan sebaran skala dan pemotongan; apakah EMG masih informatif pada rekaman-rekaman itu diukur terpisah pada I2 melalui selisih rerata EMG *Wake* dan REM.

**Temuan terkait usia.** Sebelas dari 17 rekaman EMG bermasalah berasal dari subjek berusia 80 tahun ke atas, padahal kelompok itu hanya 32 dari 153 rekaman (sekitar 21%). Sebagian lansia tampaknya memiliki sinyal EMG submental yang lemah atau berbeda. Penyebabnya belum diselidiki.

**Kesimpulan Bagian I.** Prosesnya benar, EEG dan EOG aman dipakai apa adanya, dan EMG adalah kanal yang paling tidak stabil. Normalisasi per rekaman terbukti perlu. Tidak ada rekaman yang dibuang, karena membuang subjek lansia akan mengubah populasi dan membuat hasil tidak sebanding dengan literatur. Tanda peringatan disimpan pada berkas manifes agar tahap ablasi dapat menguji sensitivitas, misalnya membandingkan hasil dengan dan tanpa 17 rekaman tersebut.

### 5.6 Bagian J: Penyimpanan *cache* dan verifikasi

Tahap ini bertujuan menyimpan hasil agar model dapat memuatnya berulang kali tanpa mengulang *pre-processing*. Analogi: porsi yang sudah ditakar dimasukkan ke kotak berlabel di kulkas. Berkas mentah di `data/raw/` tidak pernah diubah; seluruh hasil disimpan di `data/processed/sc/`.

**Kapan berkas terbentuk.** *Preprocessed record* pada `data/processed` terbentuk **hanya pada sel J1**. Fungsi yang sama dijalankan pada Bagian H, I, dan J, tetapi hanya pada J1 hasilnya ditulis ke disk (lihat tabel Bagian 5.0).

**Keputusan format.**

- **Satu berkas `.npz` per rekaman:** Pembagian lipatan nanti berdasarkan subjek, sehingga pemuatan per rekaman memudahkan perakitan lipatan tanpa memuat seluruh data.
- **Tanpa kompresi:** Sinyal biologis yang berderau tidak terkompresi dengan baik, sedangkan penulisan dan pembacaan menjadi jauh lebih lambat.
- **`float16` untuk EEG dan EOG, `float32` untuk EMG:** `float16` memangkas ukuran separuhnya. Karena nilai dibatasi ±20, presisinya memadai: galat maksimum terukur 0,0078 (terjadi pada nilai sekitar 16 sampai 20, yaitu artefak), sedangkan nilai tipikal di sekitar 1 berselisih sekitar 0,0005.

**Isi tiap berkas.** EEG (N, 2, 3.000), EOG (N, 3.000), EMG (N, 30), label per *epoch* (−1 sampai 4), nomor *epoch* asli, ID subjek, nomor malam, usia, batas jendela *trimming*, dan statistik normalisasi (median dan IQR mentah setiap kanal). ID subjek, malam, dan nomor *epoch* disimpan karena diperlukan oleh validasi berbasis subjek dan eksperimen kebocoran data (keputusan K3).

**Verifikasi (sel J2).** Seluruh berkas dibaca ulang dan diperiksa: jumlah berkas 153; bentuk dan tipe data sesuai; tidak ada nilai NaN atau tak hingga; label berada pada −1 sampai 4; nomor *epoch* berurutan; subjek dan malam sama dengan inventaris; serta jumlah per kelas seluruh berkas sama dengan hasil *trimming* (*Wake* 65.941, N1 21.522, N2 69.132, N3 13.039, REM 25.835; total 195.469). Perbandingan dengan perhitungan baru untuk SC4001E menunjukkan galat `float16` di bawah 0,01.

**Hasil.** 153 berkas dengan total 3,55 GB. Penulisan memerlukan sekitar 2,5 sampai 3 menit dan pembacaan ulang sekitar 10 sampai 13 detik. Ringkasan per rekaman (jumlah *epoch*, jumlah per kelas, statistik mutu, tanda peringatan, ukuran berkas) disimpan pada `results/tables/02_cache_manifest.csv`. Berkas ini kecil dan dapat dibagikan sehingga anggota kelompok dapat memeriksa *cache* masing-masing tanpa mengirim data.

### 5.7 Daftar gambar dan keterangan pada notebook

Gambar berikut dihasilkan notebook dan tersimpan pada `results/figures/`.

- `02_trim_window_SC4001E.png`: *hypnogram* satu malam dengan jendela yang dipertahankan dan bagian *Wake* yang dibuang.
- `02_filter_psd_SC4001E.png`: spektrum daya EEG dan EOG sebelum dan sesudah penyaringan.
- `02_emg_by_stage.png`: median dan rentang antar-kuartil EMG terolah per tahap tidur pada seluruh rekaman.
- `02_normalised_signals_by_stage_SC4001E.png`: cuplikan sinyal terolah untuk *Wake*, N3, dan REM.
- `02_qc_amplitude_and_clipping.png`: sebaran IQR mentah dan persentase pemotongan per rekaman.

Notebook `02_preprocessing` juga telah dilengkapi sel keterangan: satu sel judul, satu sel pembuka untuk tiap Bagian A sampai J, dan satu sel keterangan di bawah tiap sel kode yang memuat **Input** (apa yang dikerjakan), **Output** (apa yang tampil), dan **Kesimpulan**. Sel kode dan keluarannya tidak diubah.

---

## 6. Revisi pada Dokumen Kerangka Penelitian

Hasil Bagian E sampai J menunjukkan tiga bagian pada dokumen kerangka penelitian versi terakhir yang tidak lagi sesuai dengan implementasi. Bagian ini mencantumkan teks pengganti yang diusulkan.

### 6.1 Penanganan EMG (tiga lokasi)

**Lokasi.** *Pre-processing Pipeline* butir 3; cabang EMG pada *Feature Engineering Strategy*; dan cabang EMG pada *Epoch-level Feature Extractor* (frasa "*log-envelope*").

**Masalah.** Dokumen menyebut transformasi `log(clip(x, ε))`. Transformasi ini tidak dapat diterapkan karena 4,2% nilai EMG negatif (Bagian 5.3).

**Teks pengganti untuk butir 3:**

> Sinyal EMG submental pada SC berupa *amplitude envelope* 1 Hz (30 sampel/*epoch*) dan tidak di-*upsample*. Sebanyak 4,2% nilainya negatif, tersebar pada 119 dari 153 rekaman, sehingga transformasi logaritma tidak dapat diterapkan. Sinyal dinormalisasi per rekaman dengan *robust scaling* (median dan IQR), dipotong pada ±20 IQR, lalu ditransformasi dengan *inverse hyperbolic sine* (`arcsinh`), yang berperilaku seperti logaritma pada nilai besar tetapi terdefinisi untuk nilai negatif (Burbidge dkk., 1988). Dari empat kandidat transformasi, kombinasi ini menghasilkan distribusi paling simetris (*skewness* −0,874) dan AUC pemisah *Wake* dari REM tertinggi (0,850).

**Penyesuaian lain.** Pada kedua lokasi lainnya, ganti "*log-envelope*" dengan "*envelope* yang dinormalisasi secara *robust* dan ditransformasi `arcsinh`".

### 6.2 Penanganan label *Unscored* dan *Movement time*

**Lokasi.** *Pre-processing Pipeline* butir 2 (*AASM Label Mapping*).

**Masalah.** Dokumen menyebut kedua label tersebut dibuang, padahal keputusan K1 mempertahankannya dengan label −1.

**Teks pengganti:**

> Label *Sleep stage ?* (*Unscored*) dan *Movement time* tidak dibuang dari urutan, melainkan diberi *ignore label* (−1) sehingga dikecualikan dari perhitungan *loss* dan metrik evaluasi. *Epoch* tersebut dipertahankan agar kontinuitas jendela Bi-LSTM tidak terputus. Setelah *trimming*, terdapat 298 *epoch* berlabel −1 dari 195.767 *epoch* di dalam jendela pada SC-78. Jumlah *epoch* berlabel lima kelas (195.469) tidak berubah.

**Penambahan pada *Evaluation Matrix*.** Tambahkan kalimat: "*Loss* (termasuk *Focal Loss*) dan seluruh metrik dihitung hanya pada *epoch* berlabel 0 sampai 4 (*mask* label ≠ −1)."

### 6.3 Pernyataan tentang fitur buatan tangan

**Lokasi.** Kalimat pembuka *Feature Engineering Strategy*.

**Masalah.** Dokumen menyatakan "secara penuh meninggalkan" fitur buatan tangan, sedangkan rencana eksperimen memuat *baseline* berbasis fitur buatan tangan sebagai pembanding.

**Teks pengganti:**

> Model utama meninggalkan fitur buatan tangan dan konversi spektrogram 2D, dan mengekstraksi fitur secara otomatis menggunakan *Multi-Branch 1D-CNN* langsung dari sinyal 1D. Fitur spektral buatan tangan dengan *gradient boosting* hanya dipakai sebagai *baseline* pembanding non-*deep learning*.

### 6.4 Penambahan dan penyesuaian kecil

- **Filter (butir 4):** Tambahkan keterangan: "Penyaringan *band-pass* 0,3–35 Hz memakai filter Butterworth orde 4 yang dijalankan maju dan mundur (fase nol) sehingga morfologi gelombang tidak bergeser waktunya. Filter diterapkan pada jendela hasil *trimming* dan mendahului normalisasi per rekaman."
- **Butir baru (mutu dan *cache*):** Tambahkan butir kelima: "Seluruh 153 rekaman diperiksa mutunya sebelum disimpan. Kanal EEG dan EOG tidak menunjukkan sinyal datar dan paling banyak 0,35% sampel terpotong. Sebanyak 17 rekaman ditandai karena EMG yang hampir datar atau tidak membedakan REM dari *Wake* (11 di antaranya subjek berusia ≥ 80 tahun). Rekaman tersebut tidak dibuang dan tandanya disimpan untuk uji sensitivitas pada ablasi. Hasil disimpan sebagai satu berkas per rekaman (153 berkas, 3,55 GB) dan diverifikasi ulang: 195.469 *epoch* berlabel, sama dengan hasil *trimming*."
- **Istilah "*spatial*":** Pada *Proposed Model Architecture*, frasa "*intra-epoch spatial/morfological features*" sebaiknya diganti "*intra-epoch temporal/morphological features*", karena 1D-CNN per kanal mengekstraksi fitur temporal dan morfologi gelombang, bukan fitur spasial antar-elektroda.
- **Klaim komputasi:** Pada *Feature Engineering Strategy*, frasa "secara drastis memangkas *computational overhead*" sebaiknya diganti "menyederhanakan *pipeline* (*end-to-end* tanpa tahap transformasi terpisah)", karena transformasi spektral seperti STFT tidak mahal secara komputasi dan alasan yang lebih kuat adalah kesederhanaan *pipeline*.
- **Angka yang tidak perlu diubah:** Pada *Data Volume*, angka 237.936 *epoch* berlabel setelah *trimming* sudah konsisten dengan hasil ini (195.469 pada SC ditambah 42.467 pada ST).

---

## 7. Rencana Minggu Depan

Pada kerangka penelitian, ekstraksi fitur dilakukan secara otomatis oleh *Multi-Branch 1D-CNN*. Dengan demikian "ekstraksi fitur" pada rencana ini sama dengan *Epoch-level Feature Extractor*, yaitu bagian pertama model. Analogi: *cache* sudah tersimpan di kulkas; minggu depan dimasak dahulu menu sederhana sebagai pembanding, lalu peralatan untuk menu utama (model *deep learning*) disiapkan. Sasaran dibedakan menjadi target minimum dan target tambahan agar tidak menjanjikan melebihi yang realistis.

### 7.1 Target minimum

**1. *Baseline* klasik (belum tercantum pada kerangka penelitian; penambahannya diusulkan pada Bagian 6.3).**

- **Konsep:** Dari tiap *epoch* diambil angka ringkasan buatan tangan (misalnya kekuatan gelombang delta, alfa, dan sigma serta tingkat EMG), lalu dipelajari oleh *gradient boosting*. Fungsinya tiga: menjadi batas bawah (model *deep learning* baru layak diklaim berguna bila melampauinya), menguji kewarasan data dan pembagian lipatan, dan menjadi pembanding non-*deep learning* pada matriks ablasi.
- **Cara:** Memakai *cache* baru sehingga *baseline* dan model melihat data yang persis sama; pembagian per **subjek** (`GroupKFold`) dengan sebagian subjek latih disisihkan sebagai validasi; metrik dihitung hanya pada *epoch* berlabel 0 sampai 4.
- **Alasan pengulangan:** Hasil sebelumnya (Macro-F1 73,7% dengan validasi per subjek, dan 81,0% dengan pembagian acak per *epoch*) dihitung dari fitur tanpa penyaringan dan normalisasi.
- **Keputusan yang diperlukan:** fitur apa yang diambil, dan jumlah lipatan (misalnya 5 lipatan untuk menghemat waktu).

**2. *Dataset* dan *loader*.**

- **Konsep:** *Bi-LSTM* membaca urutan *epoch*, bukan *epoch* tunggal, sehingga data disusun sebagai jendela *L* *epoch* berurutan. Label −1 tetap berada di dalam urutan sebagai konteks tetapi tidak dihitung pada *loss*.
- **Cara:** Memuat berkas `.npz` per rekaman dan memotongnya menjadi jendela panjang *L* (awal *L* = 10 sesuai *Temporal Sequence Modeler* pada kerangka penelitian; 20 dan 30 untuk ablasi), menyiapkan *mask* label ≠ −1, dan membagi data per subjek terlebih dahulu sebelum dijendelakan agar tidak ada jendela yang melintasi dua subjek. Dimulai pada SC-20 (39 rekaman, sekitar 42 ribu *epoch*, sekitar 0,8 GB sehingga muat di memori).
- **Keputusan yang diperlukan:** nilai *L* awal, dan apakah EEG dimulai dengan satu atau dua kanal (K2).

**3. *Epoch-level Feature Extractor* (Poin 3 dan 4 kerangka penelitian).**

- **Konsep:** Tiga "pemindai" bekerja terpisah pada tiap *epoch*, lalu hasilnya digabung. Pemisahan ini mencegah sinyal beramplitudo besar menenggelamkan yang kecil.
- **Cara:** Tiga cabang 1D-CNN: EEG dengan *dual-scale kernel* (kernel kecil untuk gelombang cepat seperti *spindle*, kernel besar untuk gelombang lambat) menghasilkan 64 dimensi, EOG menghasilkan 64 dimensi, dan cabang EMG ringan atas 30 sampel menghasilkan 32 dimensi, digabung menjadi vektor sekitar 160 dimensi per *epoch*. Penyusunan dilakukan cabang demi cabang dengan pengecekan bentuk keluaran pada satu *batch*. Targetnya adalah bentuk keluaran yang terverifikasi, bukan akurasi.

### 7.2 Target tambahan (bila waktu memungkinkan)

***Temporal Sequence Modeler* dan pelatihan pertama (Poin 4 dan 5 kerangka penelitian).**

- **Konsep:** *Bi-LSTM* membaca urutan vektor fitur maju dan mundur untuk memahami konteks (89,8% *epoch* berlabel sama dengan *epoch* berikutnya). Keluarannya adalah probabilitas lima kelas per *epoch*.
- **Cara:** *Bi-LSTM* dua lapis dan satu kali pelatihan pada SC-20 dengan validasi berbasis subjek, *Cross-Entropy* dengan *mask*, *early stopping* pada Macro-F1 validasi, dan evaluasi pada subjek uji. *Focal Loss* dan pembobotan kelas diuji pada ablasi.

### 7.3 Yang akan diperbarui

| Berkas | Isi pembaruan |
|---|---|
| Notebook baru (untuk *baseline* dan untuk model) | Dikerjakan per sel dengan keterangan *Input*, *Output*, dan *Kesimpulan* seperti notebook 02 |
| `docs/belajar/` | Penjelasan sel demi sel untuk notebook baru |
| `docs/catatan/KEPUTUSAN_DESAIN.md` | Keputusan baru (K2 bila diputuskan, nilai *L*, fitur *baseline*) |
| `README.md` | Status Bagian E sampai J selesai, K4 diputuskan, estimasi waktu, dan baris notebook baru |
| Report | Tiga revisi utama pada Bagian 6 |

### 7.4 Ekspektasi dan batasan

Hasil pelatihan pertama biasanya jauh di bawah target literatur untuk SC-20 (Macro-F1 sekitar 78 sampai 82), karena belum ada penyetelan. Oleh karena itu minggu depan dijanjikan komponen yang terbukti berjalan, bukan angka akhir. Keputusan K2 (satu atau dua kanal EEG) dan K3 (eksperimen kebocoran data pada model akhir) tetap ditunda ke tahap ablasi. Perlu diingat pula bahwa SC-20 hanya berisi subjek berusia 25 sampai 34 tahun, sehingga skornya tidak setara dengan SC-78 dan tidak boleh dibandingkan langsung.

---

## 8. Keterbatasan dan Hal yang Belum Dilakukan

- **Belum ada hasil model:** Pekerjaan sejauh ini terbatas pada data. Belum ada angka Macro-F1 atau Kappa dari model apa pun pada *pipeline* yang baru.
- **Keunggulan `arcsinh` kecil:** Selisih AUC terhadap data mentah hanya 0,008; keputusan terutama didasarkan pada bentuk distribusi. Dampak sesungguhnya terhadap model baru dapat diukur pada ablasi.
- **Penyebab nilai EMG negatif belum terbukti:** Dugaan *offset* perangkat cocok dengan pola data tetapi tidak didukung dokumentasi dataset.
- **Kualitas EMG pada lansia:** Tujuh belas rekaman bermasalah tidak dibuang dan penyebab pola usianya belum diselidiki. Dampaknya terhadap hasil baru dapat dinilai pada ablasi.
- **Ambang tanda peringatan bersifat heuristik:** Ambang 1%, 5%, dan IQR di bawah 0,1 adalah ketetapan penulis, bukan standar baku.
- **Penyaringan pada jendela, bukan rekaman penuh:** Argumen bahwa efek tepi tidak menyentuh *epoch* tidur bersifat perkiraan dari lamanya transien filter (beberapa detik) dibanding lebar *Wake* di tepi jendela (30 menit), dan belum diuji dengan membandingkan kedua cara.
- **Gambar H3 dan D3 tidak sepenuhnya sebaris:** *Epoch Wake* yang dipilih berbeda (Bagian 5.4).
- **Hanya SC:** *Pipeline* ini baru mencakup rekaman SC. Rekaman ST memerlukan penyesuaian EMG (konversi ke *envelope* 1 Hz) dan baru ditangani pada tahap uji eksternal.
- **Rujukan sekunder:** Angka pembanding dari makalah lain dikutip dari sumber sekunder dan belum diperiksa satu per satu ke makalah aslinya.

---

## 9. Antisipasi Pertanyaan

- **Mengapa `arcsinh`, padahal selisih AUC terhadap data mentah hanya 0,008?** Distribusi yang lebih simetris dan terbatas lebih mudah dipelajari jaringan saraf, sedangkan tidak ada nilai yang terbuang, berbeda dengan logaritma yang membuang 4,274% nilai.
- **Mengapa EMG bernilai negatif?** Dokumentasi dataset tidak menjelaskannya. Dugaan terbaik adalah *offset* pada rangkaian perekam; kesimpulan yang aman adalah EMG tidak dapat dibaca sebagai kekuatan otot absolut dan karena itu dinormalisasi per rekaman.
- **Mengapa EMG tidak disaring?** EMG sudah berupa *envelope*, informasinya ada pada tingkat sinyal yang akan terbuang oleh *high-pass*, dan batas Nyquist-nya hanya 0,5 Hz sehingga filter 35 Hz tidak bermakna.
- **Mengapa 17 rekaman EMG bermasalah tidak dibuang?** Membuang subjek lansia mengubah populasi dan membuat hasil tidak sebanding dengan literatur. Tandanya disimpan agar sensitivitasnya dapat diuji pada ablasi.
- **Apakah normalisasi menyebabkan kebocoran data antar-*fold*?** Tidak. Statistik dihitung per rekaman dari rekaman itu sendiri, bukan dari gabungan data latih dan uji.
- **Mengapa filter dijalankan maju dan mundur?** Agar gelombang tidak bergeser waktunya, sehingga struktur seperti *K-complex* tetap berada pada *epoch* yang benar.
- **Mengapa *epoch* berlabel −1 dipertahankan, bukan dibuang?** Pembuangan akan menyambung *epoch* yang semula terpisah sehingga konteks waktu *Bi-LSTM* menjadi keliru. Dengan label −1, urutan tetap utuh dan *epoch* tersebut tidak dihitung pada *loss* maupun metrik.
- **Mengapa jumlah *Wake* berbeda dari angka pada sebagian makalah?** Aturan *trimming* berbeda antar penulis. Jumlah N1 sampai REM identik, sehingga perbedaan hanya berada pada kelas *Wake*.
- **Kapan sinyal benar-benar dipangkas dan kapan data tersimpan?** Jendela ditentukan dan label dipotong pada Bagian E; sinyal dipotong pada fungsi `preprocess_record` (Bagian H); data tertulis ke disk hanya pada Bagian J.

---

## 10. Skrip Penjelasan untuk Dosen (Versi Singkat)

Skrip ini ditulis dengan bahasa lisan yang sederhana dan sekitar tiga sampai lima menit bila dibacakan. Kalimat dalam kurung adalah petunjuk gambar yang ditampilkan.

**Pembuka.**
"Selamat siang, Pak/Bu. Saya akan menyampaikan progres minggu ini. Singkatnya, tahap persiapan data sudah selesai. Kami sudah mengubah 153 rekaman tidur menjadi data siap latih, dan semua angkanya cocok dengan hasil analisis sebelumnya dan dengan literatur. Belum ada hasil model; itu rencana minggu depan."

**Minggu lalu, singkat.**
"Minggu lalu kami sudah memahami data dan membuat label tahap tidur untuk setiap potongan 30 detik. Satu keputusan penting waktu itu: potongan yang tidak punya label yang sah, misalnya saat subjek bergerak, tidak dibuang melainkan diberi tanda khusus, supaya urutan waktunya tetap utuh dan model nanti tidak menghitungnya."

**Minggu ini, langkah demi langkah.**

"Pertama, *trimming*. Satu rekaman rata-rata 22 jam karena alatnya menyala seharian, jadi lebih dari dua pertiga isinya terjaga. Kami hanya mempertahankan masa tidur ditambah 30 menit di kedua sisinya. Hasilnya porsi terjaga turun dari 68,8% menjadi 33,7%, sedangkan data tidurnya tidak ada yang hilang. (*Tunjukkan gambar hypnogram dengan area biru.*)"

"Kedua, penyaringan. Sinyal otak dan mata disaring supaya hanya frekuensi 0,3 sampai 35 Hz yang tersisa, karena di situ informasi tahap tidur berada, sedangkan di luarnya hanya derau. Pada gambar, garis biru dan abu-abu berhimpit di bagian yang penting dan hanya berbeda di luarnya. (*Tunjukkan gambar spektrum.*)"

"Ketiga, sinyal otot atau EMG, dan ini yang paling menarik. Di rancangan awal kami berencana memakai logaritma. Setelah dicek ke semua rekaman, ternyata 4,2% nilainya negatif, sehingga logaritma tidak bisa dipakai. Kami membandingkan empat cara dan memilih normalisasi per rekaman yang diikuti transformasi arcsinh, yaitu semacam logaritma yang aman untuk nilai negatif. Kami juga tidak menyaring sinyal otot ini, karena informasinya ada pada tingkat nilainya, dan filter akan menghapusnya. Hasilnya wajar secara fisiologis: nilai otot turun dari terjaga sampai REM, sesuai kelumpuhan otot saat REM. (*Tunjukkan tabel empat kandidat dan gambar EMG per tahap.*)"

"Keempat, normalisasi. Skala sinyal tiap rekaman berbeda-beda, jadi tiap rekaman kami samakan skalanya dengan acuan yang tahan terhadap nilai ekstrem, dihitung dari rekaman itu sendiri sehingga tidak ada kebocoran antara data latih dan uji."

"Kelima, pemeriksaan mutu pada seluruh 153 rekaman. Sinyal otak dan mata bersih. Masalahnya hanya di sinyal otot: 17 rekaman ditandai, dan 11 di antaranya dari subjek berusia 80 tahun ke atas. Rekaman itu tidak kami buang supaya populasinya tidak bergeser. Tandanya kami simpan untuk diuji pengaruhnya nanti. (*Tunjukkan gambar mutu.*)"

"Terakhir, hasilnya disimpan satu berkas per rekaman, total 153 berkas atau 3,55 GB, dan sudah kami baca ulang untuk memastikan isinya benar. Jumlah potongan berlabelnya 195.469, sama persis dengan angka acuan."

**Perubahan pada report.**
"Ada tiga bagian di dokumen kerangka penelitian yang kami perbarui supaya sesuai kenyataan: cara penanganan sinyal otot, perlakuan potongan tanpa label, dan pernyataan tentang fitur buatan tangan."

**Rencana minggu depan.**
"Minggu depan target minimumnya tiga hal. Pertama, model pembanding sederhana dengan fitur buatan tangan, supaya kita punya batas bawah. Kedua, penyusun data yang membaca urutan potongan 30 detik. Ketiga, bagian pertama model, yaitu tiga cabang jaringan untuk sinyal otak, mata, dan otot. Kalau waktu cukup, kami tambahkan lapisan Bi-LSTM dan satu kali pelatihan pertama pada data 20 subjek. Kami belum menjanjikan angka akhir karena hasil pertama biasanya masih di bawah literatur."

**Penutup.**
"Kami ingin masukan Bapak/Ibu: apakah pembagian rencana ini sudah sesuai, dan apakah pembanding sederhana dengan fitur buatan tangan perlu dimasukkan ke laporan sebagai pembanding. Terima kasih."

---

## 11. Rujukan

- Burbidge, J. B., Magee, L., & Robb, A. L. (1988). Alternative transformations to handle extreme values of the dependent variable. *Journal of the American Statistical Association, 83*(401), 123–127. https://doi.org/10.1080/01621459.1988.10478575
  *Catatan penggunaan:* makalah ini membahas transformasi `arcsinh` pada variabel terikat. Pada proyek ini transformasi diterapkan pada sinyal masukan, sehingga makalah ini dikutip sebagai sumber transformasinya, bukan sebagai pembenaran penerapannya.
- Kemp, B., Zwinderman, A. H., Tuk, B., Kamphuisen, H. A. C., & Oberye, J. J. L. (2000). Analysis of a sleep-dependent neuronal feedback loop: The slow-wave microcontinuity of the EEG. *IEEE Transactions on Biomedical Engineering, 47*(9), 1185–1194. Dataset: PhysioNet *Sleep-EDF Expanded* v1.0.0, https://physionet.org/content/sleep-edfx/1.0.0/
- Rosenberg, R. S., & Van Hout, S. (2013). The American Academy of Sleep Medicine inter-scorer reliability program: Sleep stage scoring. *Journal of Clinical Sleep Medicine, 9*(1), 81–87. https://doi.org/10.5664/jcsm.2350
  *Catatan penggunaan:* kutip hanya angka kesepakatan antar-penilai (rata-rata sekitar 82,6% sampai 83% dan N1 sebesar 63,0%), karena deskripsi ukuran sampel pada abstrak dan teks lengkap tidak sepenuhnya sama.

---

## Lampiran: Glosarium Singkat

- ***Epoch*:** Potongan sinyal berdurasi 30 detik; satuan label dan satuan masukan model.
- ***Header* EDF:** Metadata rekaman (kanal, laju sampel, satuan, durasi, *prefilter*); tidak memuat nilai sinyal.
- ***Hypnogram*:** Berkas anotasi tahap tidur oleh dokter, berupa rentang waktu per tahap.
- ***Prefilter*:** Catatan pada *header* tentang filter yang sudah dipasang oleh alat perekam sebelum sinyal didigitalkan.
- ***Rectification*:** Penyearahan; nilai negatif dibalik menjadi positif.
- ***Amplitude envelope*:** Kurva kekuatan sinyal dari waktu ke waktu, tanpa getaran cepat di dalamnya.
- ***Ignore label* (−1):** Penanda *epoch* yang tetap berada di urutan tetapi tidak dihitung pada *loss* dan metrik.
- ***Trimming*:** Pemangkasan *Wake* di luar jendela dari 30 menit sebelum tidur pertama sampai 30 menit sesudah tidur terakhir.
- ***Band-pass* fase nol:** Filter yang hanya meloloskan rentang frekuensi tertentu dan dijalankan maju lalu mundur sehingga tidak menggeser waktu gelombang.
- ***Power spectral density* (PSD):** Ukuran seberapa kuat sinyal pada tiap frekuensi.
- ***Noise floor*:** Lantai derau; getaran acak yang selalu ada pada alat ukur.
- **Batas Nyquist:** Frekuensi tertinggi yang dapat diwakili suatu laju sampel, yaitu setengah laju sampel.
- **IQR:** Selisih persentil ke-75 dan ke-25; ukuran sebaran yang tahan terhadap nilai ekstrem.
- ***Robust scaling*:** Pengurangan median lalu pembagian IQR.
- ***Clipping*:** Pemotongan nilai di luar batas tertentu pada batas itu.
- **`arcsinh`:** Fungsi sinus hiperbolik invers; mirip logaritma untuk nilai besar tetapi terdefinisi untuk nilai nol dan negatif.
- ***Skewness*:** Ukuran kemiringan distribusi; 0 berarti simetris.
- **AUC:** Ukuran kemampuan memisahkan dua kelas; 0,5 setara tebakan dan 1,0 sempurna.
- ***Atonia* REM:** Kelumpuhan otot rangka selama tidur REM, tampak sebagai EMG yang sangat rendah.
- **Kontrol kualitas (QC):** Pemeriksaan data sebelum disimpan atau dipakai.
- ***Subject-wise validation*:** Pembagian data latih dan uji berdasarkan ID subjek sehingga seluruh *epoch* subjek uji tidak pernah dilihat saat pelatihan.
- **Manifes:** Tabel ringkasan per berkas *cache* untuk pemeriksaan silang.
