# Notebook 06: Ablasi pada 20 Subjek, Bahan Presentasi

Dokumen ini untuk tim yang akan mempresentasikan `notebooks/06_ablasi_sc20.ipynb`. Istilah mengikuti tabel istilah di notebook 01.

---

## 1. Pesan utama

1. **Ablasi menjawab "komponen mana yang berperan"**, dengan mengubah satu hal per percobaan terhadap model referensi (notebook 04) dan mengukur selisihnya pada subjek yang sama. Ada sembilan perbandingan, semuanya pada 20 subjek berusia 25–34 tahun, tiga *seed*.

2. **Hanya konteks waktu yang terbukti berperan.** Tanpa Bi-LSTM, MF1 turun 2,3 poin per subjek (interval 95% −3,4 sampai −1,4; *p* terkoreksi 0,001; 17 dari 20 subjek memburuk). Penurunan terbesar ada pada REM (−3,4), N1 (−6,6), dan Wake (−2,5). Jendela 10, 20, dan 30 *epoch* memberi hasil yang sama (MF1 80,1 pada ketiganya), jadi 5 menit konteks sudah cukup.

3. ***Loss* dan bobot kelas tidak mengubah mutu keseluruhan.** Keempat kombinasi ada di rentang 79,9–80,7, dan tak satu pun berbeda nyata dari referensi. Bobot kelas hanya menggeser keseimbangan: sensitivitas N1 naik dari 50,5% ke 60,4%, tetapi *precision*-nya turun dari sekitar 56% ke sekitar 49% dan sensitivitas N2 turun 2–3 poin. ***Focal loss* tidak menambah apa pun** di atas bobot kelas.

4. **Sinyal tambahan membantu pelan-pelan.** MF1 naik dari 78,3 (satu kanal EEG) ke 80,1 (semua sinyal). Pada *gradient boosting* tanpa konteks, kenaikannya jauh lebih curam (68,4 ke 76,6). Setelah koreksi, tidak ada model dengan sinyal lebih sedikit yang terbukti lebih buruk daripada model lengkap.

5. **Kebocoran data menggelembungkan MF1 sebesar 10,2 poin** pada model akhir (80,4 menjadi 90,6) dan 7,0 poin pada *baseline* (79,2 menjadi 86,2). Hanya pembagian per subjek yang sah.

## 2. Posisi dalam proyek

Notebook 04 membangun model dan notebook 05 mengujinya pada 78 subjek. Notebook ini menjawab pertanyaan yang tersisa: **mengapa model itu bekerja?** Hasilnya juga menutup dua keputusan yang sengaja ditunda sampai ada model akhir (jumlah kanal EEG dan kebocoran data) serta memisahkan dua faktor yang tercampur di notebook 04 (bentuk *loss* dan bobot kelas).

**Mengapa hanya pada 20 subjek.** Ablasi membandingkan percobaan satu sama lain pada subjek yang sama, jadi yang penting adalah selisihnya. Notebook 04 sudah punya dua hasil untuk dipakai ulang, percobaan sinyal dan kebocoran di notebook 03 juga berbasis 20 subjek, dan melatihnya pada 78 subjek akan memakan waktu sekitar dua kali lipat. Alasan terakhir ini yang terlemah. Harga yang dibayar: seluruh subjek berusia muda, dan dengan hanya 20 subjek daya ujinya rendah.

**Batas yang harus diakui.** Percobaan konfirmasi pada 78 subjek tidak dijalankan. Kesimpulan di sini hanya berlaku untuk 20 subjek ini.

## 3. Alur presentasi

| Urutan | Yang ditampilkan | Yang diucapkan | Angka kunci |
|---|---|---|---|
| 1 | Tabel daftar percobaan (A) | Setiap percobaan hanya mengubah satu hal terhadap referensi | 9 perbandingan; 20 subjek; 3 *seed* |
| 2 | Tabel uji berpasangan (G) | Hanya penghilangan Bi-LSTM yang terbukti setelah koreksi | −2,34 [−3,43; −1,37], *p* Holm 0,001 |
| 3 | Grafik konteks (I) | Konteks 10 *epoch* sudah cukup; tanpa konteks, hipnogram terlalu terpecah | 77,6 vs 80,1; 19,5 vs 10,6 pergantian per jam |
| 4 | Tabel 2×2 dan grafik sensitivitas (H) | Bobot kelas menggeser keseimbangan, bukan menaikkan mutu | N1: 50,5% ke 60,4%; MF1 80,1–80,7 |
| 5 | Grafik sinyal (J) | Model *deep learning* tahan terhadap hilangnya sinyal | 78,3 ke 80,1 vs 68,4 ke 76,6 |
| 6 | Tabel kebocoran (K) | Pembagian acak per *epoch* menggelembungkan skor | +10,2 poin (CNN), +7,0 (*baseline*) |

## 4. Penjelasan per bagian

### A. Daftar percobaan
Ada sembilan percobaan terhadap satu referensi: tiga *loss* (desain 2×2 bersama referensi), satu tanpa Bi-LSTM, dua panjang jendela, tiga tingkat sinyal, dan satu pembagian data bocor. Dua di antaranya (referensi dan *focal* berbobot) dimuat dari notebook 04 setelah pengaturannya diperiksa identik.

**Mengapa satu hal per percobaan.** Di notebook 04, "CE vs *focal*" ternyata mengubah dua hal sekaligus (bentuk *loss* dan bobot kelas). Ablasi dirancang agar kesalahan semacam itu tidak terulang.

### B–D. Data, *fold*, dan pemeriksaan
Data dan urutan *epoch* sama dengan notebook 04. Pembagian per subjek identik, dan pembagian acak per *epoch* hanya dipakai di percobaan kebocoran. Pemeriksaan otomatis memastikan tidak ada *epoch* yang ada di dua sisi pembagian pada percobaan bocor. Ukuran model berubah menurut percobaan: tanpa Bi-LSTM 331.557 parameter, satu kanal EEG 901.797, dan referensi 1.089.477.

### E. Pelatihan
Setiap percobaan dilatih 20 *fold* × 3 *seed*, sekitar 21–36 menit per *run*. Total 11,9 jam GPU untuk percobaan baru dan 1,7 jam untuk percobaan kebocoran. Proses sempat terhenti beberapa kali dan dilanjutkan dari *fold* tersimpan.

### F. Hasil keseluruhan
Skor gabungan dua kelompok: percobaan *loss* dan jendela di sekitar 80, sedangkan tanpa Bi-LSTM (77,6) dan satu kanal EEG (78,3) lebih rendah. Pembanding: *baseline* 79,2.

### G. Uji berpasangan
**Cara membacanya.** Selisih dihitung per subjek terhadap referensi, lalu dirangkum dengan interval *bootstrap*. Karena ada sembilan perbandingan, *p* dikoreksi (Holm). Hanya tanpa Bi-LSTM yang bertahan. EEG Fpz-Cz saja sempat lolos sebelum koreksi (*p* = 0,044) tetapi tidak setelahnya (0,35).

**Hal yang perlu diucapkan dengan hati-hati.** "Tidak signifikan" di sini berarti "tidak terbukti pada 20 subjek", bukan "tidak berpengaruh". Lebar interval sekitar ±1 poin, jadi selisih di bawah itu tidak mungkin terdeteksi.

### H. *Loss* dan bobot kelas
Desain 2×2: γ (0 atau 2) × bobot kelas (tidak atau ya). Efek bobot +0,37 (−0,45 sampai +1,26), efek γ = 2 −0,44 (−1,02 sampai +0,13), interaksi −0,04. Semua tidak berbeda dari nol (*p* tanpa koreksi 0,76, 0,23, 0,87).

**Temuan yang paling mudah disalahpahami.** Pada tahap N1, bobot kelas menaikkan sensitivitas (50,5% ke 60,4%) tetapi menurunkan *precision* (sekitar 56% ke 48–50%), karena model lebih sering menebak N1 (6,0% ke 8,0–8,3% dari semua tebakan, dibanding porsi sebenarnya 6,6%). Ini pergeseran ambang keputusan, bukan model yang lebih baik. **Koreksi terhadap notebook 04:** kenaikan sensitivitas N1 dan N3 yang waktu itu tampak sebagai efek *focal loss* berasal dari bobot kelas.

### I. Konteks waktu
Tanpa Bi-LSTM, prediksi berganti tahap 19,5 kali per jam, jauh lebih sering daripada pakar (12,3) dan *baseline* (15,1). Dengan Bi-LSTM angkanya 10,6–11,5. Dugaan bahwa model dengan konteks terlalu mulus belum diuji.

### J. Sinyal masukan
Pada *gradient boosting* tanpa konteks, setiap sinyal baru menaikkan MF1 1,5–4,2 poin. Pada CNN + Bi-LSTM, hanya 0,2–0,9 poin per langkah. Pembandingan ini tidak sepadan, karena *gradient boosting* di tabel tanpa konteks. Dugaannya konteks menutup sebagian informasi yang hilang. Tanpa konteks dan dengan semua sinyal, CNN (77,6) hanya sekitar 1 poin di atas *gradient boosting* (76,6). Kanal kedua EEG: +0,57 (−0,34 sampai +1,51), *p* = 0,26, tidak terbukti. EMG tidak menaikkan MF1 keseluruhan, tetapi F1-REM naik 1,8 poin.

### K. Kebocoran data
Pembagian acak per *epoch* memasukkan *epoch* tetangga dari malam yang sama ke data latih. CNN: akurasi 85,1 ke 93,3, MF1 80,4 ke 90,6, F1-N1 54,2 ke 75,4, κ 0,796 ke 0,908. Model yang lebih besar terkena lebih parah (10,2 dibanding 7,0 poin pada *baseline*). Penyebabnya masih dugaan.

## 5. Keterbatasan yang perlu diakui

1. **Daya uji rendah.** Selisih di bawah sekitar 1 poin tidak terdeteksi pada 20 subjek.
2. **Hanya usia 25–34 tahun.** Pola tidur dan mutu sinyal berubah menurut usia, jadi temuan belum tentu berlaku pada subjek lanjut usia. Percobaan konfirmasi pada 78 subjek tidak dijalankan.
3. **Satu pembagian subjek** untuk semua percobaan. Ablasi menjelaskan peran komponen, bukan memilih model.
4. **Tangga sinyal berurutan.** Kontribusi EMG diukur setelah EEG dan EOG sudah ada.
5. **Beberapa kesimpulan hanyalah dugaan:** penyebab ketahanan terhadap hilangnya sinyal, penyebab kebocoran lebih parah pada model besar, dan kecenderungan model terlalu mulus.
6. **Percobaan kebocoran hanya satu *seed*.**

## 6. Antisipasi pertanyaan

**Mengapa hanya Bi-LSTM yang signifikan?** Karena efeknya besar (sekitar 2,3 poin, 17 dari 20 subjek memburuk). Efek lain kecil atau nol, dan 20 subjek terlalu sedikit untuk membedakan efek di bawah 1 poin.

**Apakah *focal loss* tidak berguna?** Pada data ini tidak ada keuntungan tambahan dibanding bobot kelas. Hal ini belum tentu berlaku pada ketidakseimbangan yang lebih ekstrem.

**Jika bobot kelas menaikkan sensitivitas N1, mengapa tidak dipakai?** Karena skor gabungannya tidak membaik, dan *precision* turun. Pilihan ini soal kebutuhan: untuk skrining yang tidak boleh melewatkan N1, model berbobot masuk akal. Konfigurasi referensi dipertahankan karena tidak ada percobaan yang terbukti lebih baik.

**Apakah EMG tidak berguna?** Pada MF1 keseluruhan tidak terbukti. Pada F1-REM naik 1,8 poin, sejalan dengan fisiologi REM (otot melemas). Hasil ini tidak pasti pada 20 subjek.

**Mengapa jendela 10 sudah cukup?** Konteks dipakai untuk memutuskan *epoch* yang ambigu dari tetangganya. Lima menit sudah mencakup sebagian besar tetangga yang relevan. Menambah panjang tidak membantu dan membuat hasil antar-*seed* lebih bervariasi (jendela 30: simpangan baku 0,8).

**Mengapa skor 90,6 pada percobaan kebocoran tidak dilaporkan sebagai hasil?** Karena data uji dan latih berasal dari malam yang sama. Skor itu hanya untuk menunjukkan seberapa besar kebocoran menggelembungkan angka.

**Mengapa tidak diulang pada 78 subjek?** Biayanya sekitar dua kali lipat, dan konfigurasi benchmark sudah dibekukan sebelum ablasi. Hasil ini diakui terbatas pada 20 subjek.

## 7. Lampiran

**Fungsi di `src/sleep_dl.py`:** `load_records`, `loso_folds`, `epochwise_folds`, `run_cv`, `summarise`, `per_subject_mf1`, `paired`, `holm`, `SleepNet`.

**Berkas:**

| Berkas | Isi |
|---|---|
| `results/tables/06_ablation_summary.csv` | MF1, κ, F1 per tahap per percobaan |
| `results/tables/06_ablation_paired.csv` | Uji berpasangan per percobaan |
| `results/tables/06_ablation_loss_grid.csv`, `..._loss_contrasts.csv` | Desain 2×2 |
| `results/tables/06_ablation_context.csv` | Konteks dan pergantian tahap |
| `results/tables/06_ablation_modality.csv` | Tangga sinyal |
| `results/tables/06_ablation_leakage.csv` | Kebocoran data |
| `results/tables/06_ablation_per_subject.csv`, `..._runs.csv` | MF1 per subjek, per *run* |
| `results/figures/06_ablation_loss.png`, `..._context.png`, `..._modality.png` | Grafik |

**Keputusan:** K8 (rancangan dan hasil), K2 (jumlah kanal EEG), K3 (kebocoran pada model akhir).

**Rujukan:** notebook 01 (istilah), 03 (*baseline*), 04 (model referensi), 05 (benchmark 78 subjek).
