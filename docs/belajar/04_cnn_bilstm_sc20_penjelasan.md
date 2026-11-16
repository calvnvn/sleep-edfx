# Penjelasan Notebook 04_cnn_bilstm_sc20: CNN Multi-Cabang dan Bi-LSTM pada SC-20

Dokumen ini menemani `notebooks/04_cnn_bilstm_sc20.ipynb` (Fase 3, versi perombakan). Notebook tersebut melatih model *deep learning* pertama pada 20 subjek SC-20 dan membandingkannya dengan *baseline* Fase 2. Seluruh angka di dokumen ini berasal dari keluaran notebook yang telah dijalankan penuh (enam *run*, sekitar 2,8 jam pada GPU RTX 3050 6 GB). Keputusan rancangan dicatat pada K6 di `docs/catatan/KEPUTUSAN_DESAIN.md`.

---

## Bagian 1. Gambaran besar

### 1.1 Apa yang dikerjakan

**Analogi.** Pada Fase 2 kita menyusun "daftar centang" sendiri. Pada Fase 3 kita melatih "dokter muda" yang belajar langsung dari gelombang mentah, dan yang membaca rekaman seperti dokter sungguhan: tidak per 30 detik secara terpisah, melainkan sambil melihat *epoch* di sekitarnya.

**Teknis.** Satu jendela berisi 20 *epoch* berturut-turut (10 menit). Tiap *epoch* diubah menjadi satu vektor oleh tiga cabang CNN 1D (EEG, EOG, EMG). Vektor 20 *epoch* dibaca oleh Bi-LSTM dua lapis yang menelusuri urutan maju dan mundur, lalu menghasilkan 5 skor tahap tidur untuk setiap *epoch*.

```
EEG (2 kanal) ─► cabang dua skala ─► 128 ┐
EOG (1 kanal) ─► cabang dua skala ─►  64 ├─► gabung 224 ─► Bi-LSTM (20 epoch) ─► 5 skor per epoch
EMG (selubung) ─► konvolusi kecil ─►  32 ┘
```

### 1.2 Mengapa dirombak

Versi pertama Fase 3 (cabang rekan tim) menghasilkan MF1 78,6 dan tidak melampaui *baseline* (79,2), serta menurunkan Wake dan N2. Peninjauan menemukan beberapa kelemahan yang membuat hasilnya sulit ditafsirkan:

| Temuan | Akibat | Perbaikan |
|---|---|---|
| *Focal loss* menjadikan bobot kelas ikut sebagai pangkat ($p_t = p^w$) | Hukuman salah arah; hasil tidak mencerminkan *focal loss* sebenarnya | Rumus dibenahi dan diuji tiga cara |
| Tidak ada pembanding CE | Efek *focal* tak dapat dipisahkan | CE dijalankan dengan pengaturan sama |
| Validasi hanya satu subjek | Pemilihan titik henti berisik | Dua subjek, identik dengan *baseline* |
| Dua skala kernel tidak paralel | Rancangan tidak sesuai rencana | Dua jalur paralel dalam satu cabang |
| Satu kali jalan | Selisih kecil tak dapat dinilai | Tiga *seed* dan uji berpasangan |

### 1.3 Ringkasan hasil (SC-20, rerata ± simpangan baku tiga *seed*)

| | ACC | MF1 | Kappa | F1-N1 |
|---|---|---|---|---|
| *Baseline* Fase 2 | 84,1 | 79,2 | 0,784 | 49,3 |
| CNN + Bi-LSTM, CE | 85,1 ± 0,1 | 80,1 ± 0,4 | 0,795 ± 0,001 | 53,0 ± 1,1 |
| CNN + Bi-LSTM, *focal* | 84,5 ± 0,6 | 80,2 ± 0,9 | 0,790 ± 0,008 | 53,4 ± 2,4 |

**Kesimpulan singkat.** Kedua model berada sekitar satu poin MF1 di atas *baseline*, tetapi uji berpasangan per subjek tidak signifikan (Bagian 6). Yang dapat dinyatakan dengan lebih yakin: F1-N1 naik sekitar empat poin, dan penurunan Wake dan N2 pada versi pertama tidak terulang.

---

## Bagian 2. Konsep yang perlu dipahami

**Jendela urutan dan *many-to-many*.** Satu *epoch* sering ambigu bila dilihat sendirian (N1 mirip Wake dan N2). Model menerima 20 *epoch* sekaligus dan memberi 20 tebakan. Pada pengujian, jendela digeser 10 *epoch* sehingga tiap *epoch* mendapat beberapa tebakan dari jendela berbeda, yang dirata-ratakan. Jendela tidak melintasi dua rekaman.

**Dua skala kernel.** Kernel 50 sampel (0,5 detik) peka pada gelombang cepat, kernel 400 sampel (4 detik) pada gelombang lambat. Keduanya berjalan paralel pada sinyal yang sama lalu digabung, seperti dua kaca pembesar dengan perbesaran berbeda.

**Cross-entropy dan *focal loss*.** CE: hukuman = $-\alpha_y \log p_t$. *Focal*: hukuman = $-\alpha_y (1-p_t)^\gamma \log p_t$. Faktor $(1-p_t)^\gamma$ mengecilkan hukuman untuk contoh yang sudah mudah, sehingga model lebih memperhatikan contoh sulit seperti N1. $\alpha_y$ (bobot kelas) hanya pengali. Rumus *focal loss* berasal dari Lin dkk. (2017; https://doi.org/10.1109/ICCV.2017.324).

**Bobot kelas.** Berbanding terbalik dengan akar frekuensi, dinormalkan rerata 1. Pada SC-20: Wake 0,89, N1 1,52, N2 0,60, N3 1,07, REM 0,92. Pilihan ini moderat dan belum dibandingkan dengan skema lain.

**Validasi dan *seed*.** Sekitar 10% subjek latih (dua subjek) menjadi validasi untuk *early stopping* pada Macro-F1. Karena pelatihan memakai angka acak, setiap konfigurasi dijalankan dengan tiga *seed* dan selisih yang tidak melampaui variasi antar-*seed* tidak boleh diklaim sebagai perbaikan.

---

## Bagian 3. Penjelasan per bagian notebook

| Bagian | Yang dilakukan | Hasil kunci |
|---|---|---|
| A | Konfigurasi (`CONFIG`), perangkat GPU, bukti prediksi *baseline* tersedia | RTX 3050 6 GB terdeteksi |
| B | Memuat 39 rekaman subjek 0 sampai 19 | 42.368 *epoch* (42.307 berlabel); N1 hanya 2.804 (6,6%) |
| C | `SequenceDataset`: jendela 20 *epoch* dalam satu rekaman | SC4001E: 841 *epoch* menjadi 84 jendela |
| D | Arsitektur: cabang EEG, EOG, EMG, Bi-LSTM, kepala | 1.089.477 parameter (Bi-LSTM 757.760 atau sekitar 70%) |
| E | Bobot kelas dan *focal loss* dengan tiga uji | Ketiga uji lulus |
| F | Subjek validasi identik dengan notebook 03; fungsi pelatihan; uji menghafal | 16 jendela dihafal 100%, *loss* akhir 0,001 |
| G | Enam *run* × 20 *fold*; tabel ringkasan | Lihat Bagian 1.3 |
| H | Uji berpasangan, matriks kebingungan, kesalahan Wake, kurva latih, hipnogram | Bagian 4 sampai 7 |
| I | Menyimpan tabel dan prediksi | Lima tabel `04_cnn_sc20_*.csv` |

### 3.1 Tiga uji *focal loss* (Bagian E)

1. Dengan $\gamma = 0$, *focal loss* harus sama dengan CE berbobot. Bila tidak, galat ada pada kerangka rumusnya.
2. Nilainya harus cocok dengan perhitungan tangan pada contoh kecil.
3. Baris berlabel −1 tidak boleh memengaruhi nilai *loss*.

Ketiganya lulus, sehingga bila *focal* berbeda dari CE, penyebabnya bukan galat rumus.

### 3.2 Mengapa ada uji menghafal (Bagian F)

Model yang kodenya benar harus mampu menghafal 16 jendela. Bila gagal, ada galat pada bentuk data, label, *loss*, atau gradien. Uji ini hanya membuktikan kodenya tidak rusak, bukan bahwa modelnya bagus.

### 3.3 Subjek validasi (Bagian F)

Subjek validasi dipilih dengan `GroupShuffleSplit` (`random_state = SEED + fold`) dan diperiksa identik dengan pilihan notebook 03 pada seluruh 20 *fold*. Contoh: subjek uji 0 divalidasi oleh subjek 2 dan 11. Dengan demikian selisih terhadap *baseline* tidak dapat berasal dari perbedaan data validasi.

---

## Bagian 4. Uji berpasangan per subjek (H1)

MF1 dihitung per subjek (rerata tiga *seed*) lalu dibandingkan secara berpasangan dengan uji peringkat bertanda Wilcoxon.

| Pasangan | Selisih rerata (poin) | Subjek lebih baik | p |
|---|---|---|---|
| CE vs *baseline* | +0,64 | 12 dari 20 | 0,475 |
| *Focal* vs *baseline* | +0,56 | 14 dari 20 | 0,202 |
| *Focal* vs CE | −0,07 | 11 dari 20 | 0,927 |

**Pembacaan.** Tidak ada selisih yang signifikan, dan dengan hanya 20 subjek uji ini bertenaga rendah. Maka: (a) belum ada bukti bahwa model *deep learning* lebih baik daripada *baseline* di SC-20; (b) tidak ada bukti bahwa *focal* berbeda dari CE. Efeknya heterogen antar-subjek: subjek 2 naik dari 72,0 ke 82,6 (CE), subjek 11 turun dari 71,0 ke 63,2 (*focal*), dan subjek 5 turun dari 84,0 ke 79,6 (CE).

---

## Bagian 5. Ke mana tiap tahap salah dibawa (H2) dan kesalahan Wake (H3)

Sensitivitas per tahap (%):

| | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| *Baseline* | 87,5 | 55,3 | 85,0 | 86,1 | 87,3 |
| CE | 89,4 | 50,5 | 88,1 | 81,2 | 89,0 |
| *Focal* | 87,9 | 60,1 | 84,9 | 85,0 | 88,6 |

- CE lebih tepat pada kelas besar tetapi lebih lemah pada N1 dan N3 (23% N1 dan 18% N3 dibawa ke N2).
- *Focal* mengembalikan N1 dan N3 tetapi Wake dan N2 kembali ke tingkat *baseline*. Artinya *focal* menukar kapasitas antar-kelas, dan MF1 hampir tidak berubah.
- **Penurunan Wake dan N2 versi pertama tidak terulang**: tidak ada model dengan Wake atau N2 di bawah *baseline*. Karena *loss*, validasi, dan arsitektur diubah bersamaan, penyebab penurunan versi pertama tidak dapat dipastikan. Dugaan galat *focal loss* konsisten dengan hasil tetapi belum diuji secara terpisah.

Galat Wake (%) menurut posisi dalam malam:

| | Sebelum tidur (2.340) | Tengah malam (3.605) | Setelah tidur (2.339) |
|---|---|---|---|
| *Baseline* | 4,8 | 23,7 | 2,9 |
| CE | 3,9 | 18,6 | 4,9 |
| *Focal* | 4,6 | 22,5 | 3,5 |

Sekitar 83% kesalahan Wake *baseline* terjadi di tengah malam (terbangun singkat). CE mengurangi galat tersebut tetapi menambah galat setelah tidur. Mekanismenya belum diuji; tabel ini hanya menunjukkan letak kesalahan.

---

## Bagian 6. Kurva pelatihan dan hipnogram (H4, H5)

- **Putaran terbaik** median 6 (rentang 1 sampai 30 dari maksimal 30). Model mencapai kinerja validasi terbaiknya dalam beberapa putaran lalu mulai menghafal subjek latih. Hanya satu *fold* (CE, *seed* 0, subjek 17) menyentuh batas 30, sehingga batas ini tidak membatasi hasil. Kendala utama adalah sedikitnya subjek latih (17).
- **Hipnogram SC4001E:** kesepakatan dengan dokter 86,2% (*baseline*) dan 86,7% (CNN, CE). Ini ilustrasi satu rekaman, bukan bukti.

---

## Bagian 7. Kesimpulan dan batasan

**Yang dapat dinyatakan.**

1. Versi perombakan berjalan benar secara teknis (semua uji lulus) dan stabil antar-*seed* (simpangan baku MF1 CE 0,4).
2. Pada SC-20, MF1 CE 80,1 ± 0,4 dan *focal* 80,2 ± 0,9 dibanding *baseline* 79,2; F1-N1 naik sekitar empat poin.
3. Penurunan Wake dan N2 versi pertama tidak terulang.

**Yang tidak dapat dinyatakan.**

1. Bahwa *deep learning* lebih baik daripada *baseline* pada SC-20: p ≥ 0,20 pada semua uji.
2. Bahwa CE lebih baik daripada *focal* (atau sebaliknya): keduanya seri pada MF1; pemilihan CE oleh validasi (82,5 vs 82,4) adalah seri praktis, dipertahankan karena lebih sederhana dan stabil.
3. Penyebab pasti penurunan Wake dan N2 versi pertama.

**Batasan.** Hanya SC-20 (20 subjek); *baseline* hanya satu kali jalan (variasi antar-*seed* tidak diukur); panjang jendela, bobot kelas, dan hiperparameter tidak disetel; belum ada SC-78 maupun ST. Langkah berikutnya (Fase 4: SC-78) hanya bermakna setelah ditentukan cabang Fase 3 mana yang dipertahankan bersama rekan tim.

---

## Lampiran. Berkas yang dihasilkan

| Berkas | Isi |
|---|---|
| `results/tables/04_cnn_sc20_runs.csv` | Metrik tiap *run* (6 baris) |
| `results/tables/04_cnn_sc20_metrics.csv` | Ringkasan rerata ± simpangan baku dan *baseline* |
| `results/tables/04_cnn_sc20_per_subject.csv` | MF1 per subjek (*baseline*, CE, *focal*) |
| `results/tables/04_cnn_sc20_paired_tests.csv` | Uji Wilcoxon berpasangan |
| `results/tables/04_cnn_sc20_wake_errors.csv` | Galat Wake menurut posisi |
| `results/figures/04_cnn_sc20_*.png` | Per subjek, matriks kebingungan, kurva latih, hipnogram |
| `data/processed/cnn_sc20_runs/`, `cnn_sc20_predictions.npz` | Prediksi tiap *run* dan model terpilih (diabaikan git) |
