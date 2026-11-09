# Pembaruan Progres: Tahap *Pre-processing* (Bagian E sampai J) dan Rencana Berikutnya

**Proyek:** *Automated Sleep Stage Classification using Multimodal Hybrid Deep Learning*
**Mata kuliah:** *Deep Learning* (AI706), Informatika, Universitas Multimedia Nusantara, Kelompok 8
**Disusun:** 9 November 2026

Dokumen ini ditulis agar dapat dibaca sendiri tanpa dokumen lain. Bagian 1 sampai 3 memberi konteks, Bagian 4 merangkum pembaruan minggu lalu, Bagian 5 menjelaskan secara rinci pekerjaan minggu ini (Bagian E sampai J pada notebook `02_preprocessing`), Bagian 6 mencatat revisi yang perlu dilakukan pada dokumen kerangka penelitian, dan Bagian 7 menguraikan rencana minggu depan. Seluruh angka dalam dokumen ini berasal dari keluaran notebook yang telah dijalankan pada seluruh 153 rekaman, bukan dari literatur, kecuali ditandai dengan sitasi.

---

## 1. Ringkasan

Minggu ini *pipeline pre-processing* untuk seluruh rekaman *Sleep-Cassette* (SC) telah selesai dibangun, diuji, dan dijalankan. Hasilnya adalah satu berkas data siap latih untuk setiap rekaman (153 berkas, total 3,55 GB) yang memuat 195.469 *epoch* berlabel, sama persis dengan angka acuan pada tahap analisis data eksplorasi (EDA) dan literatur. Terdapat empat hasil utama:

- **Pemangkasan jendela tidur (*trimming*):** *Epoch Wake* turun dari 285.433 menjadi 65.941, sehingga porsi *Wake* turun dari 68,8% menjadi 33,7%, sedangkan jumlah N1, N2, N3, dan REM tidak berubah.
- **Keputusan penanganan EMG (K4):** Transformasi logaritma pada EMG tidak dapat dipakai karena 4,2% nilai EMG negatif. Pilihan akhirnya adalah *robust scaling* per rekaman, pemotongan nilai ekstrem pada ±20, lalu transformasi `arcsinh`.
- **Pemeriksaan mutu:** Kanal EEG dan EOG bersih pada seluruh rekaman, sedangkan 17 rekaman ditandai karena kualitas EMG yang rendah (11 di antaranya berasal dari subjek berusia 80 tahun ke atas).
- **Verifikasi:** Setiap berkas dibaca ulang dan diperiksa; jumlah *epoch* per kelas sama dengan hasil *trimming*.

Belum ada hasil model apa pun (baik *deep learning* maupun *baseline*). Tahap berikutnya adalah ekstraksi fitur dan pelatihan model, yang direncanakan pada Bagian 7.

---

## 2. Konteks Proyek

### 2.1 Tujuan dan data

Proyek ini mengklasifikasikan lima tahap tidur menurut standar AASM, yaitu *Wake*, N1, N2, N3, dan REM, dari sinyal polisomnografi. Satuan data adalah *epoch* berdurasi 30 detik. Model yang diusulkan adalah *Multi-Branch 1D-CNN* (cabang EEG, EOG, dan EMG) yang digabung dengan *Bi-LSTM* dua lapis dan dilatih dengan *Focal Loss*, dievaluasi memakai validasi berbasis subjek (*subject-wise GroupKFold*) dengan *Macro-F1* dan *Cohen's Kappa* sebagai metrik utama.

Dataset yang dipakai adalah PhysioNet *Sleep-EDF Expanded* v1.0.0 (Kemp dkk., 2000), yang terdiri atas 197 rekaman dalam dua kelompok:

- **Sleep-Cassette (SC):** 153 rekaman dari 78 subjek sehat berusia 25 sampai 101 tahun, direkam di rumah. Kelompok ini menjadi data utama pelatihan dan evaluasi (disebut SC-78). Subset 20 subjek pertama (39 rekaman, usia 25 sampai 34 tahun) disebut SC-20 dan dipakai untuk iterasi cepat.
- **Sleep-Telemetry (ST):** 44 rekaman dari 22 subjek dengan kesulitan tidur ringan dalam studi efek *Temazepam*. Kelompok ini hanya dipakai sebagai uji generalisasi di luar distribusi pelatihan (*out-of-distribution*) dan tidak dibahas pada dokumen ini.

### 2.2 Istilah dasar

- **Rekaman PSG dan *hypnogram*:** Setiap rekaman terdiri atas satu berkas sinyal (`*-PSG.edf`) dan satu berkas anotasi dokter (`*-Hypnogram.edf`) yang mencatat tahap tidur sebagai rentang waktu.
- **Kanal yang dipakai:** EEG Fpz-Cz dan EEG Pz-Oz serta EOG horizontal, semuanya pada 100 Hz (3.000 sampel per *epoch*), dan EMG submental. EMG pada SC bukan sinyal mentah, melainkan *amplitude envelope* 1 Hz (30 sampel per *epoch*), yakni nilai kekuatan otot per detik hasil penyearahan dan penghalusan oleh alat perekam.
- **Label −1 (*ignore label*):** Keputusan K1 menetapkan bahwa *epoch* berlabel *Sleep stage ?* (*Unscored*) dan *Movement time* tidak dibuang dari urutan, melainkan diberi nilai −1 sehingga dikecualikan dari perhitungan *loss* dan metrik. *Epoch* tersebut tetap berfungsi sebagai konteks waktu bagi *Bi-LSTM*, yang membaca urutan *epoch* berkelanjutan.

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

## 4. Pembaruan Minggu Lalu (Ringkas)

Minggu lalu pekerjaan berfokus pada pemahaman data dan pembentukan label yang benar.

- **EDA:** Dataset diverifikasi langsung dari berkas EDF. Dihitung rata-rata waktu tiap tahap tidur dan dibuat visualisasi kondisi data (distribusi kelas sebelum dan sesudah *trimming*, durasi rekaman, komposisi tahap menurut usia). Ditemukan bahwa EMG pada SC berupa *envelope* 1 Hz, bukan sinyal mentah, sehingga tidak boleh di-*upsample* maupun disaring *band-pass*.
- **Pembentukan label (notebook 02, Bagian A sampai D):** PSG dan *hypnogram* dipasangkan memakai tujuh karakter pertama nama berkas (153 pasangan dari 78 subjek). Rentang waktu pada *hypnogram* diubah menjadi satu label per *epoch*, dipotong mengikuti panjang sinyal, dan hasilnya dicocokkan dengan hasil EDA pada seluruh rekaman (416.424 *epoch*, dengan 414.961 berlabel dan 1.463 berlabel −1).
- **Keputusan K1:** Label −1 untuk *epoch Unscored* dan *Movement time*, sebagaimana dijelaskan pada Bagian 2.2.

---

## 5. Pembaruan Minggu Ini: Bagian E sampai J

Pekerjaan minggu ini merangkai tahap-tahap berikut menjadi satu fungsi per rekaman, lalu menjalankannya pada seluruh 153 rekaman: pemangkasan jendela tidur, penyaringan, penanganan EMG, normalisasi, pemeriksaan mutu, dan penyimpanan. Urutan tahap penting. Penyaringan dilakukan sebelum normalisasi karena filter mengubah amplitudo, dan normalisasi dilakukan setelah *trimming* agar statistiknya tidak didominasi data siang hari.

### 5.1 Bagian E: *Trimming* jendela tidur

Tahap ini bertujuan membuang bagian rekaman yang terjaga di luar waktu tidur. Rekaman SC rata-rata berdurasi 22,7 jam karena alat dibiarkan menyala sepanjang hari, sehingga *Wake* mencapai 68,8% dari seluruh *epoch* berlabel. Sebagian besar *Wake* itu terjadi saat subjek beraktivitas biasa, tidak informatif untuk membedakan tahap tidur, dan membuat kelas sangat tidak seimbang.

**Aturan.** *Epoch* tidur didefinisikan sebagai *epoch* berlabel N1, N2, N3, atau REM. *Wake* dan label −1 tidak dihitung sebagai tidur. Jendela yang dipertahankan dimulai 60 *epoch* (30 menit) sebelum *epoch* tidur pertama dan berakhir 60 *epoch* sesudah *epoch* tidur terakhir. Batas jendela dijaga agar tidak melewati ujung rekaman. *Epoch* berlabel −1 di dalam jendela tidak dibuang, sesuai keputusan K1.

**Contoh satu rekaman (SC4001E).** *Epoch* tidur pertama berada pada nomor 1.021 dan yang terakhir pada nomor 1.741, sehingga jendelanya mencakup *epoch* 961 sampai 1.801 (841 *epoch*, atau 7,0 jam dari 22,1 jam). Jumlah *Wake* turun dari 1.997 menjadi 188, sedangkan N1, N2, N3, dan REM tidak berubah. Ini sifat penting *trimming*: ia hanya mengurangi *Wake* dan tidak pernah menyentuh *epoch* tidur.

**Hasil pada seluruh 153 rekaman.** Jendela seluruh rekaman memuat 195.767 *epoch*, terdiri atas 195.469 *epoch* berlabel dan 298 *epoch* berlabel −1.

| Kelas | Sebelum *trimming* | Sesudah *trimming* | Porsi sebelum | Porsi sesudah |
|---|---|---|---|---|
| *Wake* | 285.433 | 65.941 | 68,8% | 33,7% |
| N1 | 21.522 | 21.522 | 5,2% | 11,0% |
| N2 | 69.132 | 69.132 | 16,7% | 35,4% |
| N3 | 13.039 | 13.039 | 3,1% | 6,7% |
| REM | 25.835 | 25.835 | 6,2% | 13,2% |

**Verifikasi.** Dua lapis uji dijalankan dan keduanya lolos. Pertama, hitungan lima kelas pada setiap rekaman dibandingkan dengan hasil EDA yang dihitung oleh fungsi terpisah. Kedua, hitungan agregat dibandingkan dengan angka acuan literatur (N1 = 21.522, N2 = 69.132, N3 = 13.039, REM = 25.835) dan dengan angka 298 *epoch* berlabel −1 pada keputusan K1. Jumlah *Wake* (65.941) berbeda dari 69.824 yang dilaporkan sebagian publikasi. Perbedaan itu bersumber dari aturan *trimming* yang berbeda antar penulis, bukan dari kekeliruan, dan jumlah N1 sampai REM yang identik menguatkannya. Karena itu aturan di atas ditulis secara eksplisit agar hasil dapat direproduksi.

### 5.2 Bagian F: Penyaringan *band-pass* untuk EEG dan EOG

Tahap ini bertujuan menyisakan rentang frekuensi yang membawa informasi tahap tidur. Ciri tahap tidur terletak pada gelombang delta (0,5 sampai 4 Hz), theta, alfa, *sleep spindle* (sekitar 11 sampai 16 Hz), dan beta rendah. Di atas sekitar 35 Hz energi sinyal didominasi derau otot dan listrik, sedangkan di bawah 0,3 Hz hanya terdapat pergeseran garis dasar (*drift*).

**Mekanisme.** Dipakai filter Butterworth orde 4 dengan pita lolos 0,3 sampai 35 Hz, yang dijalankan **maju dan mundur** (*zero-phase*). Penyaringan dua arah menghasilkan fase nol sehingga gelombang tidak bergeser waktunya, hal yang penting bagi struktur seperti *K-complex* yang harus tetap berada pada *epoch* yang benar. Konsekuensinya, gain filter dikuadratkan sehingga redaman dalam desibel menjadi dua kali lipat. Filter dirancang dalam bentuk *second-order sections* dan dijalankan dalam presisi `float64` demi stabilitas numerik. Filter diterapkan pada jendela hasil *trimming*, bukan pada 24 jam penuh, sehingga komputasinya sekitar tiga sampai empat kali lebih cepat. Transien filter pada ujung jendela diperkirakan hanya bertahan beberapa detik di dalam 30 menit *Wake* sehingga tidak menyentuh *epoch* tidur.

**Karakteristik terukur.** Gain filter pada beberapa frekuensi adalah sebagai berikut.

| Frekuensi (Hz) | 0,05 | 0,1 | 0,3 | 1 | 10 | 30 | 35 | 40 | 45 |
|---|---|---|---|---|---|---|---|---|---|
| Gain (dB) | −124,8 | −76,6 | −6,0 | 0,0 | 0,0 | −0,5 | −6,0 | −31,7 | −81,5 |

Pada rentang 1 sampai 10 Hz sinyal tidak berubah, pada tepi pita (0,3 dan 35 Hz) amplitudo menjadi setengahnya (−6 dB), dan di luar tepi redaman naik sangat cepat. Pada spektrum daya rekaman SC4001E, kurva sebelum dan sesudah penyaringan berhimpit pada rentang 0,5 sampai 30 Hz dan berpisah hanya di luar pita lolos.

**Catatan.** Perangkat SC sudah memasang *high-pass* 0,5 Hz dan *low-pass* 100 Hz pada kanal EEG dan EOG (tercatat pada *header* EDF). Oleh karena itu efek nyata filter ini terutama *low-pass* 35 Hz, sedangkan batas bawah 0,3 Hz berfungsi menjaga sisa *drift* tidak lolos. **EMG tidak disaring**, karena nilai EMG pada SC adalah *envelope* yang informasinya terletak pada tingkat (level) dan komponen sangat lambat; *high-pass* akan membuang level tonus otot itu sendiri.

### 5.3 Bagian G: Penanganan EMG (keputusan K4)

Tahap ini bertujuan memilih cara yang tepat untuk mengolah EMG sebelum dimasukkan ke cabang EMG. Rancangan awal pada kerangka penelitian memakai transformasi `log(clip(x, ε))`. Pemeriksaan pada seluruh 153 rekaman menunjukkan rancangan itu bermasalah.

**Masalah yang terukur.**

- **Nilai negatif:** 4,2% seluruh sampel EMG di dalam jendela *trimming* bernilai negatif, tersebar pada 119 dari 153 rekaman. Sebuah *envelope* secara teori tidak pernah negatif; temuan ini berasal dari *offset* kalibrasi perangkat. Transformasi logaritma tidak terdefinisi untuk nilai tersebut.
- **Skala antar-rekaman:** Median EMG per rekaman berkisar dari −0,99 sampai 17,79, dan IQR dari 0,07 sampai 5,15, sehingga nilai antar-rekaman tidak sebanding tanpa normalisasi.

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

Tahap ini bertujuan menyeragamkan skala setiap kanal agar model belajar membedakan tahap tidur, bukan membedakan perangkat atau pemasangan elektroda. Amplitudo mentah sangat bervariasi: pada SC4001E, IQR mentah EEG Fpz-Cz adalah 19,67 µV, EEG Pz-Oz 12,61 µV, dan EOG 17,95 µV.

**Aturan.**

- **Robust scaling:** Setiap kanal pada setiap rekaman dikurangi median lalu dibagi IQR (persentil 75 dikurangi persentil 25). Berbeda dari *z-score* yang memakai rerata dan simpangan baku, median dan IQR tidak mudah terdistorsi oleh satu artefak besar.
- **Pemotongan (*clipping*):** Nilai di luar ±20 IQR dipotong pada batas itu. Rentang ini sangat longgar (setara sekitar ±27 simpangan baku pada data normal) sehingga hanya artefak ekstrem yang terkena. Persentase nilai yang terpotong dicatat untuk pemeriksaan mutu.
- **Urutan:** Statistik dihitung per rekaman (per malam) **setelah** *trimming* dan penyaringan. Bila dihitung pada 24 jam penuh, statistiknya didominasi *Wake* siang hari. Karena statistik dihitung per rekaman, tidak ada informasi yang bocor antar-subjek maupun antar-*fold*.
- **EMG:** Setelah *robust scaling* dan pemotongan, EMG ditransformasi dengan `arcsinh` (Bagian 5.3).

**Keluaran per rekaman.** Seluruh langkah dirangkai dalam satu fungsi yang menghasilkan: EEG berbentuk (N, 2, 3.000) untuk dua kanal Fpz-Cz dan Pz-Oz, EOG (N, 3.000), EMG (N, 30), label (N), serta nomor *epoch* asli (N). Dua kanal EEG disimpan agar pilihan satu atau dua kanal dapat diuji sebagai ablasi tanpa mengulang *pre-processing* (keputusan K2 ditunda ke tahap pemodelan).

**Hasil pada SC4001E.** Setelah normalisasi, median EEG dan EOG tepat 0 dan IQR tepat 1,000, sedangkan IQR EMG 0,961 karena `arcsinh` sedikit memampatkan nilai. Nilai absolut maksimum EEG hanya 8,4 dan 12,4, sedangkan EOG mencapai 20,0 (tepat di batas pemotongan), yang menunjukkan adanya artefak yang terpotong.

### 5.5 Bagian I: Pemeriksaan mutu seluruh 153 rekaman

Tahap ini bertujuan menemukan rekaman bermasalah sebelum data ditulis ke disk, karena satu rekaman rusak yang lolos akan memengaruhi setiap eksperimen berikutnya. Seluruh *pipeline* dijalankan pada 153 rekaman (sekitar 130 detik), dan hanya statistik per rekaman yang dicatat pada tahap ini.

**Uji dan tanda peringatan.** Tiga uji dijalankan: seluruh nilai terhingga, jumlah *epoch* berlabel dan jumlah per kelas sama dengan hasil *trimming*, serta panjang jendela setiap rekaman sama dengan hasil *trimming*. Tiga tanda peringatan dipakai:

- **Pemotongan berlebih:** lebih dari 1% sampel pada satu kanal terpotong.
- **EEG datar:** EEG hampir konstan pada lebih dari 5% *epoch* (indikasi elektroda lepas).
- **EMG bermasalah:** IQR mentah EMG di bawah 0,1 (nyaris datar), atau rerata EMG saat REM tidak lebih rendah daripada saat *Wake*.

**Hasil.** Seluruh nilai terhingga dan seluruh hitungan cocok. Kanal EEG dan EOG bersih: tidak ada rekaman dengan EEG datar, dan persentase pemotongan terbesar hanya 0,35% (EEG Pz-Oz) serta 0,17% (EOG). Sebanyak 20 rekaman ditandai, **seluruhnya karena EMG**: 9 rekaman karena pemotongan EMG dan 17 rekaman karena kualitas EMG (ada yang tumpang tindih). Dari 17 rekaman EMG bermasalah, 6 memiliki IQR di bawah 0,1 dan 13 memiliki rerata REM yang tidak lebih rendah dari *Wake*.

**Temuan terkait usia.** Sebelas dari 17 rekaman EMG bermasalah berasal dari subjek berusia 80 tahun ke atas, padahal kelompok itu hanya 32 dari 153 rekaman (sekitar 21%). Sebagian lansia tampaknya memiliki sinyal EMG submental yang lemah atau berbeda, sehingga cabang EMG perlu dievaluasi dengan hati-hati pada kelompok usia ini.

**Variasi antar-rekaman.** IQR mentah EEG dan EOG berbeda antar-rekaman hingga 4,5 sampai 7,9 kali, dan EMG hingga 73,5 kali. Temuan ini menjadi pembenaran empiris untuk normalisasi per rekaman.

**Keputusan.** Tidak ada rekaman yang dibuang. Membuang subjek lansia akan mengubah populasi dan membuat hasil tidak sebanding dengan literatur. Tanda peringatan disimpan pada berkas manifes agar tahap ablasi dapat menguji sensitivitas, misalnya membandingkan hasil dengan dan tanpa 17 rekaman tersebut.

### 5.6 Bagian J: Penyimpanan *cache* dan verifikasi

Tahap ini bertujuan menyimpan hasil agar model dapat memuatnya berulang kali tanpa mengulang *pre-processing*. Berkas mentah di `data/raw/` tidak pernah diubah; seluruh hasil disimpan di `data/processed/sc/`.

**Keputusan format.**

- **Satu berkas `.npz` per rekaman:** Pembagian lipatan nanti berdasarkan subjek, sehingga pemuatan per rekaman memudahkan perakitan lipatan tanpa memuat seluruh data.
- **Tanpa kompresi:** Sinyal biologis yang berderau tidak terkompresi dengan baik, sedangkan penulisan dan pembacaan menjadi jauh lebih lambat.
- **`float16` untuk EEG dan EOG, `float32` untuk EMG:** `float16` memangkas ukuran separuhnya. Karena nilai dibatasi ±20, presisinya memadai: galat maksimum terukur 0,0078 (terjadi pada nilai sekitar 16 sampai 20, yaitu artefak), sedangkan nilai tipikal di sekitar 1 berselisih sekitar 0,0005.

**Isi tiap berkas.** EEG (N, 2, 3.000), EOG (N, 3.000), EMG (N, 30), label per *epoch* (−1 sampai 4), nomor *epoch* asli, ID subjek, nomor malam, usia, batas jendela *trimming*, dan statistik normalisasi (median dan IQR mentah setiap kanal). ID subjek, malam, dan nomor *epoch* disimpan karena diperlukan oleh validasi berbasis subjek dan eksperimen kebocoran data (keputusan K3).

**Verifikasi.** Seluruh berkas dibaca ulang dan diperiksa: jumlah berkas 153; bentuk dan tipe data sesuai; tidak ada nilai NaN atau tak hingga; label berada pada −1 sampai 4; nomor *epoch* berurutan; subjek dan malam sama dengan inventaris; serta jumlah per kelas seluruh berkas sama dengan hasil *trimming* (*Wake* 65.941, N1 21.522, N2 69.132, N3 13.039, REM 25.835; total 195.469). Perbandingan dengan perhitungan baru untuk SC4001E menunjukkan galat `float16` di bawah 0,01.

**Hasil.** 153 berkas dengan total 3,55 GB. Penulisan memerlukan sekitar 157 detik dan pembacaan ulang sekitar 13 detik. Satu kali jalan seluruh notebook (Bagian A sampai J) memerlukan sekitar 5,5 menit. Ringkasan per rekaman (jumlah *epoch*, jumlah per kelas, statistik mutu, tanda peringatan, ukuran berkas) disimpan pada `results/tables/02_cache_manifest.csv`. Berkas ini kecil dan dapat dibagikan sehingga anggota kelompok dapat memeriksa *cache* masing-masing tanpa mengirim data.

### 5.7 Daftar gambar pendukung

Gambar berikut dihasilkan notebook dan tersimpan pada `results/figures/`.

- `02_trim_window_SC4001E.png`: *hypnogram* satu malam dengan jendela yang dipertahankan dan bagian *Wake* yang dibuang.
- `02_filter_psd_SC4001E.png`: spektrum daya EEG dan EOG sebelum dan sesudah penyaringan.
- `02_emg_by_stage.png`: median dan rentang antar-kuartil EMG terolah per tahap tidur pada seluruh rekaman.
- `02_normalised_signals_by_stage_SC4001E.png`: cuplikan sinyal terolah untuk *Wake*, N3, dan REM.
- `02_qc_amplitude_and_clipping.png`: sebaran IQR mentah dan persentase pemotongan per rekaman.

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

Pada kerangka penelitian, ekstraksi fitur dilakukan secara otomatis oleh *Multi-Branch 1D-CNN*. Dengan demikian "ekstraksi fitur" pada rencana ini sama dengan *Epoch-level Feature Extractor*, yaitu bagian pertama model. Sasaran dibedakan menjadi target minimum dan target tambahan agar tidak menjanjikan melebihi yang realistis.

### 7.1 Target minimum

- **Baseline klasik (belum tercantum pada kerangka penelitian; penambahannya diusulkan pada Bagian 6.3):** Mengulang eksperimen fitur spektral buatan tangan dengan *gradient boosting* pada *cache* baru, memakai validasi berbasis subjek. Hasil sebelumnya (Macro-F1 73,7% dengan validasi per subjek, dan 81,0% dengan pembagian acak per *epoch*) dihitung dari fitur tanpa penyaringan dan normalisasi. Pengulangan menghasilkan angka pembanding pada data yang sama persis dengan model *deep learning*, sekaligus menjadi uji kewarasan data.
- ***Dataset* dan *loader*:** Menyusun jendela *L* *epoch* berurutan dari *cache* (*L* = 10 sebagai konfigurasi awal, sesuai *Temporal Sequence Modeler* pada kerangka penelitian), dengan label −1 tetap berada dalam urutan sebagai konteks dan di-*mask* saat perhitungan *loss*.
- ***Epoch-level Feature Extractor* (Poin 3 dan 4 kerangka penelitian):** Tiga cabang 1D-CNN (EEG dengan *dual-scale kernel*, EOG, dan cabang EMG ringan atas 30 sampel) yang masing-masing menghasilkan vektor fitur (target 64, 64, dan 32 dimensi) lalu digabung. Targetnya adalah bentuk keluaran yang terverifikasi, bukan akurasi.

### 7.2 Target tambahan (bila waktu memungkinkan)

- ***Temporal Sequence Modeler* dan pelatihan pertama (Poin 4 dan 5 kerangka penelitian):** Menambahkan *Bi-LSTM* dua lapis dan satu kali pelatihan pada SC-20 dengan validasi berbasis subjek. Pelatihan pertama memakai *Cross-Entropy* terlebih dahulu; *Focal Loss* dan pembobotan kelas diuji pada ablasi.

### 7.3 Ekspektasi dan batasan

Hasil pelatihan pertama biasanya jauh di bawah target literatur untuk SC-20 (Macro-F1 sekitar 78 sampai 82), karena belum ada penyetelan. Oleh karena itu minggu depan dijanjikan komponen yang terbukti berjalan, bukan angka akhir. Keputusan K2 (satu atau dua kanal EEG) dan K3 (eksperimen kebocoran data pada model akhir) tetap ditunda ke tahap ablasi. Perlu diingat pula bahwa SC-20 hanya berisi subjek berusia 25 sampai 34 tahun, sehingga skornya tidak setara dengan SC-78 dan tidak boleh dibandingkan langsung.

---

## 8. Keterbatasan dan Hal yang Belum Dilakukan

- **Belum ada hasil model:** Pekerjaan sejauh ini terbatas pada data. Belum ada angka Macro-F1 atau Kappa dari model apa pun pada *pipeline* yang baru.
- **Keunggulan `arcsinh` kecil:** Selisih AUC terhadap data mentah hanya 0,008; keputusan terutama didasarkan pada bentuk distribusi. Dampak sesungguhnya terhadap model baru dapat diukur pada ablasi.
- **Kualitas EMG pada lansia:** Tujuh belas rekaman bermasalah tidak dibuang. Dampaknya terhadap hasil belum diuji dan baru dapat dinilai pada ablasi.
- **Penyaringan pada jendela, bukan rekaman penuh:** Argumen bahwa efek tepi tidak menyentuh *epoch* tidur bersifat perkiraan dari lamanya transien filter (beberapa detik) dibanding lebar *Wake* di tepi jendela (30 menit), dan belum diuji dengan membandingkan kedua cara.
- **Hanya SC:** *Pipeline* ini baru mencakup rekaman SC. Rekaman ST memerlukan penyesuaian EMG (konversi ke *envelope* 1 Hz) dan baru ditangani pada tahap uji eksternal.
- **Rujukan sekunder:** Angka pembanding dari makalah lain dikutip dari sumber sekunder dan belum diperiksa satu per satu ke makalah aslinya.

---

## 9. Antisipasi Pertanyaan

- **Mengapa `arcsinh`, padahal selisih AUC terhadap data mentah hanya 0,008?** Distribusi yang lebih simetris dan terbatas lebih mudah dipelajari jaringan saraf, sedangkan tidak ada nilai yang terbuang, berbeda dengan logaritma yang membuang 4,274% nilai.
- **Mengapa 17 rekaman EMG bermasalah tidak dibuang?** Membuang subjek lansia mengubah populasi dan membuat hasil tidak sebanding dengan literatur. Tandanya disimpan agar sensitivitasnya dapat diuji pada ablasi.
- **Apakah normalisasi menyebabkan kebocoran data antar-*fold*?** Tidak. Statistik dihitung per rekaman dari rekaman itu sendiri, bukan dari gabungan data latih dan uji.
- **Mengapa filter dijalankan maju dan mundur?** Agar gelombang tidak bergeser waktunya, sehingga struktur seperti *K-complex* tetap berada pada *epoch* yang benar.
- **Mengapa *epoch* berlabel −1 dipertahankan, bukan dibuang?** Pembuangan akan menyambung *epoch* yang semula terpisah sehingga konteks waktu *Bi-LSTM* menjadi keliru. Dengan label −1, urutan tetap utuh dan *epoch* tersebut tidak dihitung pada *loss* maupun metrik.
- **Mengapa jumlah *Wake* berbeda dari angka pada sebagian makalah?** Aturan *trimming* berbeda antar penulis. Jumlah N1 sampai REM identik, sehingga perbedaan hanya berada pada kelas *Wake*.

---

## 10. Rujukan

- Burbidge, J. B., Magee, L., & Robb, A. L. (1988). Alternative transformations to handle extreme values of the dependent variable. *Journal of the American Statistical Association, 83*(401), 123–127. https://doi.org/10.1080/01621459.1988.10478575
  *Catatan penggunaan:* makalah ini membahas transformasi `arcsinh` pada variabel terikat. Pada proyek ini transformasi diterapkan pada sinyal masukan, sehingga makalah ini dikutip sebagai sumber transformasinya, bukan sebagai pembenaran penerapannya.
- Kemp, B., Zwinderman, A. H., Tuk, B., Kamphuisen, H. A. C., & Oberye, J. J. L. (2000). Analysis of a sleep-dependent neuronal feedback loop: The slow-wave microcontinuity of the EEG. *IEEE Transactions on Biomedical Engineering, 47*(9), 1185–1194. Dataset: PhysioNet *Sleep-EDF Expanded* v1.0.0, https://physionet.org/content/sleep-edfx/1.0.0/
- Rosenberg, R. S., & Van Hout, S. (2013). The American Academy of Sleep Medicine inter-scorer reliability program: Sleep stage scoring. *Journal of Clinical Sleep Medicine, 9*(1), 81–87. https://doi.org/10.5664/jcsm.2350
  *Catatan penggunaan:* kutip hanya angka kesepakatan antar-penilai (rata-rata sekitar 82,6% sampai 83% dan N1 sebesar 63,0%), karena deskripsi ukuran sampel pada abstrak dan teks lengkap tidak sepenuhnya sama.

---

## Lampiran: Glosarium Singkat

- ***Epoch*:** Potongan sinyal berdurasi 30 detik; satuan label dan satuan masukan model.
- ***Hypnogram*:** Berkas anotasi tahap tidur oleh dokter, berupa rentang waktu per tahap.
- ***Trimming*:** Pemangkasan *Wake* di luar jendela dari 30 menit sebelum tidur pertama sampai 30 menit sesudah tidur terakhir.
- ***Band-pass* fase nol:** Filter yang hanya meloloskan rentang frekuensi tertentu dan dijalankan maju lalu mundur sehingga tidak menggeser waktu gelombang.
- ***Amplitude envelope*:** Kurva kekuatan sinyal dari waktu ke waktu, tanpa getaran cepat di dalamnya.
- **IQR:** Selisih persentil ke-75 dan ke-25; ukuran sebaran yang tahan terhadap nilai ekstrem.
- ***Robust scaling*:** Pengurangan median lalu pembagian IQR.
- ***Clipping*:** Pemotongan nilai di luar batas tertentu pada batas itu.
- **`arcsinh`:** Fungsi sinus hiperbolik invers; mirip logaritma untuk nilai besar tetapi terdefinisi untuk nilai nol dan negatif.
- ***Skewness*:** Ukuran kemiringan distribusi; 0 berarti simetris.
- **AUC:** Ukuran kemampuan memisahkan dua kelas; 0,5 setara tebakan dan 1,0 sempurna.
- ***Atonia* REM:** Kelumpuhan otot rangka selama tidur REM, tampak sebagai EMG yang sangat rendah.
- ***Subject-wise validation*:** Pembagian data latih dan uji berdasarkan ID subjek sehingga seluruh *epoch* subjek uji tidak pernah dilihat saat pelatihan.
- **Manifes:** Tabel ringkasan per berkas *cache* untuk pemeriksaan silang.
