# Penjelasan Notebook 03_baseline: Fitur Buatan Tangan dan *Gradient Boosting*

Dokumen ini menemani `notebooks/03_baseline.ipynb` (Fase 2). Notebook tersebut membuat model pembanding sederhana sebelum model *deep learning* dibangun pada Fase 3. Seluruh angka di dokumen ini berasal dari keluaran notebook yang telah dijalankan pada *cache* hasil notebook 02.

---

## Bagian 1. Gambaran besar

### 1.1 Apa yang dikerjakan Fase 2

**Analogi.** Sebelum melatih seorang dokter muda yang belajar sendiri dari ribuan rekaman (Fase 3), kita membuat "perawat dengan daftar centang". Kita yang menentukan ciri apa yang diperiksa ("seberapa banyak gelombang lambat?", "apakah otot lemas?"), sedangkan mesin hanya belajar menimbang ciri-ciri itu. Bila dokter muda kelak tidak lebih baik daripada perawat ini, kerumitannya tidak terbukti bermanfaat.

**Teknis.** Setiap *epoch* diubah menjadi 33 angka ringkasan (fitur), ditambah fitur dua *epoch* sebelum dan sesudahnya, lalu dipelajari oleh *gradient boosting*. Validasi dilakukan berbasis subjek.

```
cache (.npz) ─► 33 fitur per epoch ─► + konteks ±2 (165 kolom) ─► gradient boosting ─► 5 kelas
```

Fase 2 memiliki tiga fungsi:

- **Batas bawah:** angka yang harus dilampaui model *deep learning*.
- **Uji kewarasan:** bila model sederhana pun gagal, kemungkinan ada masalah pada data atau pembagian lipatan.
- **Pembanding non-*deep learning*** pada matriks ablasi (E0).

### 1.2 Perbedaan dengan Fase 3

| | Fase 2 (notebook ini) | Fase 3 |
|---|---|---|
| Pemilih fitur | Kita, berdasarkan ilmu tidur dan pemrosesan sinyal | Jaringan saraf, belajar dari data |
| Masukan model | Tabel 33 angka per *epoch* (165 dengan konteks) | Sinyal terolah (3.000 dan 30 titik per *epoch*) |
| Konteks waktu | Fitur tetangga ditempelkan | *Bi-LSTM* membaca urutan |
| Model | *Gradient boosting* | CNN tiga cabang + *Bi-LSTM* |
| Perangkat | CPU, sekitar 25 menit | GPU disarankan |

### 1.3 Ringkasan hasil

| | ACC | MF1 | Kappa | F1-N1 |
|---|---|---|---|---|
| SC-20 (20 lipatan, satu subjek per lipatan) | 84,1 | 79,2 | 0,784 | 49,3 |
| SC-78 (10 lipatan) | 80,7 | 76,5 | 0,739 | 51,0 |

### 1.4 Konsep kunci

Istilah berikut muncul berulang di Bagian F sampai H. Ringkasan yang sama ada pada sel markdown di awal notebook.

**Pembagian per subjek.** Data disusun per *epoch* tetapi dibagi per **subjek**, karena satu subjek punya dua malam yang harus berada di sisi yang sama (latih atau uji). Dengan begitu model diuji pada orang yang belum pernah dilihatnya.

**Fold (lipatan).** Satu putaran pelatihan dan pengujian dengan satu kelompok subjek sebagai penguji. Putaran diulang bergiliran sampai setiap kelompok pernah menjadi penguji satu kali. SC-20 memakai 20 *fold* (satu subjek per *fold*) dan SC-78 memakai 10 *fold* (sekitar 7 sampai 8 subjek per *fold*). Manfaatnya: hasil tidak bergantung pada satu pembagian yang kebetulan mudah atau sulit, dan seluruh data pernah diuji tanpa pernah diuji oleh model yang sudah melihatnya. Harganya: model dilatih sebanyak jumlah *fold*.

**Data latih, validasi, dan uji.** Analogi: soal latihan, ujian tiruan, dan ujian sebenarnya. Data **validasi** (sekitar 10% subjek dari data latih) dipakai untuk memutuskan kapan pelatihan berhenti, sedangkan data **uji** hanya dinilai dan tidak boleh memengaruhi keputusan apa pun, agar nilai akhir jujur.

**Penghentian dini (*early stopping*).** Makin banyak pohon ditambahkan, kesalahan pada data latih terus turun, tetapi kesalahan pada data validasi akhirnya berhenti turun lalu naik karena model mulai menghafal. Pelatihan dihentikan bila *log-loss* validasi tidak membaik selama 20 iterasi berturut-turut. Validasi diambil per subjek agar ujian tiruannya mirip kondisi nyata.

**Metrik.** ACC adalah persentase tebakan benar dan bisa menyesatkan pada kelas tidak seimbang; MF1 adalah rata-rata F1 kelima kelas dengan bobot sama (metrik utama); Kappa adalah kesepakatan dengan dokter setelah dikoreksi kesepakatan kebetulan; F1-N1 adalah F1 khusus kelas tersulit.

---

## Bagian 2. Penjelasan sel per sel

### Sel A1. Konfigurasi

**Isi.** Pustaka, lokasi berkas, dan parameter tetap: label −1 diabaikan (keputusan K1), konteks dua *epoch*, *seed* 0, serta jumlah *epoch* per kelas yang diharapkan dari notebook 02 (*Wake* 65.941, N1 21.522, N2 69.132, N3 13.039, REM 25.835).

**Mengapa ada `assert`.** Notebook ini membaca *cache* dari notebook 02. Bila 153 berkas tidak ada, notebook berhenti dengan pesan jelas.

### Sel B1. Fungsi fitur

**Analogi.** Daftar centang perawat: setiap baris daftar adalah satu ciri yang dicari dokter saat menilai tahap tidur.

**Dasar pemilihan fitur.** Fitur tidak dikarang, tetapi berasal dari dua sumber:

1. **Ciri yang dipakai dokter (aturan AASM):** gelombang alfa saat *Wake*, theta saat N1, *sleep spindle* saat N2, gelombang lambat saat N3, serta gerakan mata cepat dan otot lumpuh saat REM.
2. **Rumus pemrosesan sinyal yang baku:** spektrum Welch, parameter Hjorth, dan kurtosis.

**Daftar fitur.**

| Kanal | Jumlah | Fitur |
|---|---|---|
| EEG Fpz-Cz dan EEG Pz-Oz (masing-masing) | 12 | Daya total 0,5–30 Hz (log); daya relatif enam pita (log): delta lambat 0,5–2 Hz, delta 2–4 Hz, theta 4–8 Hz, alfa 8–12 Hz, sigma 12–15 Hz, beta 15–30 Hz; rasio theta/alfa dan delta/beta (log); *Hjorth mobility* dan *complexity*; kurtosis |
| EOG | 5 | Daya gerakan lambat 0,3–2 Hz dan cepat 2–8 Hz (log), rasio keduanya, amplitudo puncak-ke-puncak (log), kurtosis |
| EMG | 4 | Rerata, simpangan baku, minimum, dan maksimum dari 30 nilai |

**Cara menghitung daya pita.** Grafik spektrum (PSD) pada notebook 02 menunjukkan kekuatan sinyal pada tiap frekuensi. Daya sebuah pita adalah luas di bawah kurva itu pada rentang frekuensinya. Spektrum dihitung sekali per kanal dengan metode Welch (jendela 4 detik, tumpang tindih 50%), lalu dijumlahkan per pita. Daya relatif adalah daya pita dibagi daya total, sehingga tidak bergantung pada skala rekaman. Logaritma dipakai karena daya tersebar sangat lebar.

**Fitur lain.**

- ***Hjorth mobility*:** perkiraan frekuensi rata-rata sinyal; rendah saat gelombang lambat (N3).
- ***Hjorth complexity*:** seberapa berubah-ubah frekuensinya.
- **Kurtosis:** seberapa "runcing" sebaran nilai; tinggi bila ada lonjakan sesaat seperti kedipan mata atau *K-complex*.

**Keluaran.** SC4001E menghasilkan 841 *epoch* × 33 fitur, semuanya terhingga. Tabel median memperlihatkan daya gelombang lambat tertinggi saat N3 dan EMG terendah saat REM.

### Sel C1. Ekstraksi fitur seluruh rekaman

**Cara kerja.** Fungsi yang sama dijalankan pada 153 berkas *cache*. Hasilnya digabung menjadi satu tabel dan disimpan ke `data/processed/features_sc.npz` beserta label, rekaman, subjek, malam, usia, dan nomor *epoch*.

**Keluaran.** 195.767 *epoch* × 33 fitur dalam sekitar 2 sampai 4 menit. Tiga `assert` memastikan tidak ada nilai kosong, jumlah per kelas sama dengan notebook 02, dan 298 *epoch* berlabel −1 ikut tersimpan. *Epoch* −1 disimpan karena tetap dipakai sebagai konteks bagi tetangganya.

### Sel D1. Pemeriksaan fitur per tahap tidur

**Tujuan.** Fitur yang benar harus mengikuti fisiologi tidur. Bila tidak, perhitungannya keliru.

**Cara membaca gambar.** Setiap panel adalah satu fitur. Titik adalah median dan batang adalah rentang antar-kuartil untuk tiap tahap.

| Fitur | Pola yang terlihat | Makna |
|---|---|---|
| Daya gelombang lambat relatif (Fpz-Cz) | Tertinggi saat N3 | Gelombang lambat khas tidur dalam |
| Daya alfa relatif (Pz-Oz) | Tertinggi saat *Wake* | Ritme alfa saat terjaga, terkuat di kanal oksipital |
| *Hjorth mobility* (Fpz-Cz) | Terendah saat N3 | Frekuensi rata-rata turun saat gelombang lambat |
| Daya sigma relatif (Fpz-Cz) | **Tidak** tertinggi saat N2 | N1 tertinggi (−3,76), N2 −3,91, batangnya tumpang tindih; N3 paling rendah (−5,19). Lihat catatan sigma di bawah |
| Amplitudo EOG | Tertinggi saat *Wake* dan REM | Kedipan dan gerakan mata cepat |
| Tingkat EMG | Turun berurutan dari *Wake* ke REM | *Atonia* REM |

**Catatan tentang sigma.**

- **Apa itu sigma.** Pita 12 sampai 15 Hz tempat *sleep spindle* muncul, yaitu ledakan gelombang singkat yang khas tahap N2. Karena itu daya sigma diharapkan paling tinggi saat N2.
- **Apa yang ditemukan.** Median sigma tertinggi justru saat N1 (−3,76, setara sekitar 2,3% dari total daya), sedangkan N2 −3,91 (sekitar 2,0%), dan batang keduanya tumpang tindih. Jadi fitur ini **tidak membedakan N2 dari N1**. Sigma tetap berguna untuk memisahkan N3, yang jauh lebih rendah (−5,19).
- **Dugaan penyebab (belum diuji).**
  1. *Spindle* hanya berlangsung sekitar 0,5 sampai 2 detik, kira-kira 3% dari satu *epoch*, sehingga ledakannya tenggelam saat daya dirata-ratakan selama 30 detik. Durasi *spindle* ini berasal dari pengetahuan umum dan belum diperiksa ke sumber tertentu.
  2. Sigma yang dipakai adalah daya **relatif**, yaitu dibagi daya total. Saat N3 daya total didominasi gelombang lambat yang besar sehingga bagian sigma tampak kecil, sedangkan saat N1 gelombang lambat belum banyak sehingga bagian sigma tampak lebih besar. Sebagian selisih bisa berasal dari penyebut, bukan dari sigma itu sendiri.
  3. Batas pita 12–15 Hz adalah pilihan kita; sebagian penulis memakai 11–16 Hz.
- **Cara memastikan.** Ubah batas pita menjadi 11–16 Hz (percobaan kecil nomor 3), hitung daya sigma mutlak, atau gambar satu *epoch* N2 untuk melihat apakah *spindle* tampak sebagai ledakan singkat.

**Kesimpulan.** Lima dari enam fitur berperilaku sesuai harapan, sehingga perhitungannya dapat dipercaya. Tidak ada fitur yang cukup sendirian karena banyak batang yang tumpang tindih, sehingga model menggabungkan banyak fitur. Bila dugaan tentang sigma benar, model yang belajar mendeteksi ledakan singkat sendiri (Fase 3) berpotensi lebih baik, tetapi itu juga baru dugaan.

### Sel E1. Fitur konteks

**Analogi.** Dokter tidak menilai satu *epoch* sendirian; ia melihat juga apa yang terjadi beberapa menit sebelum dan sesudahnya.

**Cara kerja.** Untuk setiap *epoch*, fitur dua *epoch* sebelum dan dua sesudahnya ditempelkan sehingga satu baris berisi 33 × 5 = 165 kolom. `np.roll` menggeser tabel; baris yang tetangganya berasal dari rekaman lain diisi NaN. *Gradient boosting* dapat menangani nilai kosong, sehingga *epoch* di tepi rekaman tetap dapat dipakai tanpa nilai palsu.

**Uji.** Jumlah nilai kosong tepat 30.294 = 153 rekaman × 6 posisi tepi (1 + 2 di awal, 1 + 2 di akhir) × 33 fitur. Satu *epoch* contoh juga dicek: kolom "+1" harus sama dengan fitur *epoch* berikutnya.

### Sel F1. Model, metrik, dan skema validasi

**Analogi model.** *Gradient boosting* seperti sekelompok murid yang menjawab bergiliran: murid kedua khusus memperbaiki kesalahan murid pertama, murid ketiga memperbaiki sisanya, dan seterusnya. Setiap "murid" adalah pohon keputusan kecil.

**Pengaturan model.**

- `HistGradientBoostingClassifier` dari scikit-learn (tidak perlu pustaka baru).
- Laju belajar 0,1, maksimum 31 daun per pohon, `max_bins = 63`.
- `class_weight = "balanced"`: kelas jarang (N1, N3) diberi bobot lebih besar.
- **Penghentian dini:** pelatihan berhenti bila *log-loss* pada subjek validasi tidak membaik selama 20 iterasi (maksimum 500).

**Analogi validasi.** Ujian harus memakai orang yang belum pernah dilihat model, bukan sekadar malam yang belum dilihat.

**Skema validasi.**

1. Data dibagi per **subjek** menjadi lipatan latih dan uji.
2. Di dalam data latih, sekitar 10% subjek disisihkan sebagai **validasi** untuk penghentian dini. Data uji tidak pernah dipakai untuk mengambil keputusan.
3. *Epoch* berlabel −1 tidak dilatih dan tidak dinilai.
4. `assert` memastikan tidak ada subjek di dua himpunan sekaligus.
5. Metrik dihitung dari prediksi gabungan seluruh lipatan.

**Metrik.**

- ***Macro-F1* (MF1):** rata-rata F1 lima kelas dengan bobot sama; metrik utama karena tidak tertipu kelas mayoritas.
- ***Cohen's Kappa*:** kesepakatan model dengan dokter setelah dikoreksi kesepakatan kebetulan.
- **F1 per kelas**, terutama N1, dan **akurasi**.

### Sel G1 dan G2. Hasil utama

**Protokol.** SC-20 memakai satu subjek per lipatan (20 lipatan) dan SC-78 memakai 10 lipatan, mengikuti protokol yang lazim di literatur.

| | MF1 | Kappa | Rerata MF1 per lipatan | Waktu |
|---|---|---|---|---|
| SC-20 | 79,2 | 0,784 | 78,6 ± 5,8 | sekitar 3 menit |
| SC-78 | 76,5 | 0,739 | 76,4 ± 2,4 | sekitar 11 menit |

Rentang per lipatan SC-20 lebih lebar karena satu lipatan hanya berisi satu subjek. SC-78 lebih rendah karena memuat subjek lanjut usia (Sel H4).

### Sel G3. Tabel dan matriks konfusi

| | ACC | MF1 | F1 Wake | F1 N1 | F1 N2 | F1 N3 | F1 REM | Kappa |
|---|---|---|---|---|---|---|---|---|
| SC-20 | 84,1 | 79,2 | 89,8 | 49,3 | 86,8 | 83,6 | 86,5 | 0,784 |
| SC-78 | 80,7 | 76,5 | 91,8 | 51,0 | 81,5 | 74,6 | 83,5 | 0,739 |

**Cara membaca matriks konfusi.** Setiap baris adalah tahap sebenarnya dan setiap kolom tahap tebakan. Angka adalah persentase per baris: dari seluruh *epoch* tahap tertentu, berapa persen ditebak sebagai tahap apa. Diagonal adalah tebakan benar.

**Temuan.**

- N1 hanya benar 55% (SC-20) dan 60% (SC-78). Sisanya tertebak sebagai N2 (17% dan 20%), *Wake* (14% dan 11%), dan REM (13% dan 7%).
- N3 tertebak sebagai N2 sekitar 13 sampai 14%, wajar karena keduanya sama-sama tidur NREM dan batasnya bertahap.
- Pada SC-78, 12% *epoch* N2 tertebak N1. Ini sebagian akibat bobot kelas yang mendorong model lebih sering menebak N1.

**Kesimpulan.** N1 adalah kelas tersulit, sesuai literatur dan kesepakatan antar-dokter yang rendah untuk N1 (sekitar 63%; Rosenberg & Van Hout, 2013). Skor ini sudah dekat dengan beberapa model *deep learning* di literatur (misalnya DeepSleepNet MF1 76,9 pada SC-20), meski protokolnya tidak persis sama sehingga perbandingan langsung harus hati-hati.

### Sel H1. Ablasi modalitas (SC-20)

**Cara kerja.** Model dilatih ulang dengan masukan yang ditambah bertahap, memakai protokol yang sama dengan G1.

| Masukan | Fitur | MF1 | F1-N1 | F1-REM |
|---|---|---|---|---|
| EEG Fpz-Cz | 12 | 68,4 | 29,3 | 68,4 |
| + EEG Pz-Oz | 24 | 72,5 | 34,6 | 73,9 |
| + EOG | 29 | 75,1 | 41,7 | 79,1 |
| + EMG | 33 | 76,6 | 43,7 | 82,9 |
| + konteks ±2 (model utama) | 165 | 79,2 | 49,3 | 86,5 |

**Kesimpulan.** Setiap modalitas menambah skor. EOG dan EMG paling membantu REM, sesuai fisiologinya, dan konteks antar-*epoch* memberi tambahan terbesar kedua. Ini bukti awal untuk desain multimodal dan *Bi-LSTM* pada Fase 3.

### Sel H2. Pemeriksaan kebocoran data

**Analogi.** Ujian yang soalnya mirip sekali dengan soal latihan: nilainya tinggi, tetapi tidak mengukur kemampuan sebenarnya.

**Cara kerja.** Model yang sama dilatih dengan pembagian **acak per *epoch***, sehingga *epoch* dari orang yang sama bisa berada di data latih dan uji sekaligus. Hasilnya dibandingkan dengan pembagian per subjek dari G1.

**Mengapa pembagian acak menggelembungkan skor.** Ada dua jalur. Pertama, model bisa **mengenali orangnya** dari ciri sinyal khas orang itu. Kedua, **tetangga kembar**: bila *epoch* 500 masuk data uji dan *epoch* 499 serta 501 (hampir identik, label biasanya sama) masuk data latih, maka baris *epoch* 501 memuat fitur *epoch* 500 pada kolom `@-1` lengkap dengan labelnya. Jalur kedua adalah penalaran yang belum diuji terpisah.

**Hasil.**

| Pembagian | ACC | MF1 | F1-N1 | Kappa |
|---|---|---|---|---|
| Per subjek (benar) | 84,1 | 79,2 | 49,3 | 0,784 |
| Acak per *epoch* (bocor) | 90,0 | 86,2 | 63,6 | 0,863 |

MF1 naik 7,0 poin dan F1-N1 naik 14,3 poin hanya karena cara membagi data. Selisih ini konsisten dengan percobaan lama sebelum *pre-processing* (73,7 vs 81,0).

**Kesimpulan.** Kenaikan itu palsu karena model mengenali orangnya, bukan tahap tidurnya. Ini alasan seluruh evaluasi proyek wajib berbasis subjek. Eksperimen yang sama akan diulang pada model akhir (keputusan K3).

### Sel H3. Pentingnya fitur (SC-78)

**Cara kerja (*permutation importance*).** Setelah model dilatih pada satu lipatan, satu kelompok kolom pada data uji diacak antar-*epoch*, sehingga hubungannya dengan label rusak. Penurunan MF1 menunjukkan seberapa besar model bergantung pada kelompok itu. Setiap fitur diacak bersama kelima salinan konteksnya.

**Hasil.**

| Modalitas | Penurunan MF1 bila diacak |
|---|---|
| EEG Fpz-Cz | 21,6 poin |
| EEG Pz-Oz | 19,8 poin |
| EOG | 11,2 poin |
| EMG | 1,9 poin |

Fitur tunggal terpenting adalah daya total Fpz-Cz (5,5 poin), diikuti kurtosis Fpz-Cz, kurtosis EOG, dan daya delta 2–4 Hz Pz-Oz.

**Mengapa EMG tampak kecil.** Pengacakan hanya mengukur informasi yang **tidak** dimiliki fitur lain. Sebagian informasi EMG tumpang tindih dengan EOG dan EEG, sehingga ketika EMG diacak, model masih bisa bersandar pada fitur lain. Ablasi H1 tetap menunjukkan EMG menambah 1,5 poin MF1 dan 3,8 poin F1-REM. Kedua ukuran saling melengkapi dan tidak saling bertentangan.

### Sel H4. Skor per subjek menurut usia (SC-78)

| Kelompok usia | Subjek | Rerata MF1 | Simpangan baku |
|---|---|---|---|
| 25–39 | 20 | 79,0 | 4,7 |
| 40–59 | 21 | 73,2 | 6,7 |
| 60–79 | 21 | 71,2 | 10,4 |
| 80 ke atas | 16 | 68,7 | 8,3 |

**Cara membaca gambar.** Setiap titik adalah satu subjek; titik oranye adalah subjek yang EMG-nya ditandai bermasalah pada notebook 02.

**Temuan.** Korelasi usia dengan MF1 per subjek −0,43. Subjek ber-EMG bermasalah rata-rata 68,8, dibanding 74,3 untuk lainnya, tetapi selisih ini sebagian besar tumpang tindih dengan efek usia karena kebanyakan dari mereka lansia.

**Kesimpulan.** Model makin sulit pada subjek lanjut usia, sejalan dengan berkurangnya N3 dan bertambahnya fragmentasi tidur pada usia tua. Hasil proyek perlu dilaporkan terstratifikasi usia.

### Sel I1. Penyimpanan

Tabel tersimpan di `results/tables/` dengan awalan `03_baseline_` (metrik, ablasi, kebocoran, pentingnya fitur, dan skor per subjek). Prediksi SC-20 dan SC-78 tersimpan di `data/processed/baseline_predictions.npz` (diabaikan git) agar dapat dibandingkan langsung dengan prediksi Fase 3 per *epoch*.

**Batas bawah Fase 3:** MF1 79,2 / Kappa 0,784 pada SC-20 dan MF1 76,5 / Kappa 0,739 pada SC-78.

---

## Bagian 3. Istilah yang muncul

| Istilah | Arti dalam notebook ini |
|---|---|
| Fitur | Angka ringkasan satu *epoch*, misalnya daya alfa |
| Daya pita | Luas di bawah kurva spektrum pada satu rentang frekuensi |
| Daya relatif | Daya pita dibagi daya total |
| Parameter Hjorth | *Mobility* (frekuensi rata-rata) dan *complexity* (perubahan frekuensi) |
| Kurtosis | Ukuran keruncingan sebaran; tinggi bila ada lonjakan sesaat |
| Fitur konteks | Fitur *epoch* tetangga yang ditempelkan ke sebuah baris |
| *Gradient boosting* | Gabungan banyak pohon keputusan kecil yang saling memperbaiki |
| Penghentian dini | Menghentikan pelatihan saat skor validasi tidak lagi membaik |
| *Log-loss* | Ukuran kesalahan probabilitas; makin kecil makin baik |
| `class_weight="balanced"` | Bobot kelas berbanding terbalik dengan frekuensinya |
| `LeaveOneGroupOut` | Satu subjek menjadi data uji pada setiap lipatan |
| `GroupKFold` | Pembagian lipatan yang menjaga satu subjek tetap di satu lipatan |
| *Permutation importance* | Penurunan skor ketika satu kelompok kolom diacak |
| Ablasi | Mengurangi atau menambah satu komponen untuk mengukur kontribusinya |

## Bagian 4. Percobaan kecil

1. Ubah `CONTEXT` menjadi 0 dan 4, jalankan ulang Bagian E sampai G1, dan bandingkan MF1 SC-20.
2. Hapus `class_weight="balanced"` pada fungsi `model()` dan perhatikan perubahan F1-N1 dan F1-*Wake*.
3. Ubah batas pita sigma menjadi 11–16 Hz dan lihat apakah daya sigma relatif saat N2 menjadi lebih menonjol pada Bagian D.
4. Pada H1, tambahkan konfigurasi "EEG Fpz-Cz + EMG" (tanpa EOG) dan bandingkan F1-REM-nya dengan "+ EOG".

## Bagian 5. Pertanyaan untuk menguji pemahaman

1. Apa perbedaan mendasar antara fitur pada Fase 2 dan "fitur" yang dipelajari CNN pada Fase 3?
2. Mengapa daya pita dinyatakan relatif terhadap daya total, bukan daya mutlak?
3. Mengapa tetangga dari rekaman lain diisi NaN, bukan nol?
4. Mengapa subjek validasi diambil dari data latih, bukan dari data uji?
5. Mengapa pembagian acak per *epoch* menghasilkan skor lebih tinggi, dan mengapa skor itu tidak boleh dilaporkan?
6. Mengapa EMG tampak kecil pada *permutation importance* tetapi tetap menambah skor pada ablasi?
7. Mengapa skor per subjek menurun seiring usia?
