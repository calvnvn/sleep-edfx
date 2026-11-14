# Catatan Keputusan Desain

Berkas ini mencatat keputusan rancangan yang diambil selama pengerjaan, beserta alasan dan dampaknya pada Ref-1 dan Fase 1. Tiap keputusan memiliki status: **Diputuskan**, **Ditunda**, atau **Terbuka**.

## K1. Penanganan epoch *Unscored* dan *Movement time* di dalam jendela *trimming*

**Status:** Diputuskan (30 Oktober 2026).

**Keputusan.** Epoch berlabel *Sleep stage ?* (*Unscored*) dan *Movement time* yang masih berada di dalam jendela *trimming* **tidak dibuang dari urutan**. Sinyalnya dipertahankan, labelnya diberi nilai khusus (−1), dan epoch tersebut **dikecualikan dari perhitungan *loss* maupun metrik** (MF1, κ, matriks konfusi).

**Alasan.**

1. Bi-LSTM membaca jendela L epoch berurutan. Jika epoch di tengah malam dibuang lalu sisanya disambung, epoch yang semula terpisah menjadi tampak bersebelahan sehingga konteks waktunya keliru.
2. Dengan nilai −1, urutan waktu tetap utuh dan model tetap dapat memakai epoch tersebut sebagai konteks bagi epoch di sekitarnya.
3. Secara efektif epoch itu tetap "dibuang" dari penilaian, sesuai maksud Ref-1.

**Dasar angka** (diukur dari hypnogram, setelah *trimming* ±30 menit).

| | SC-78 | ST |
|---|---|---|
| Epoch berlabel *?* atau *Movement time* di dalam jendela | 298 (±0,15% dari 195.767) | 1.690 (±3,8% dari 44.157) |
| Di tengah malam (di antara tidur pertama dan terakhir) | 287 | 82 |
| Di tepi jendela (sebelum tidur pertama atau sesudah tidur terakhir) | 11 | 1.608 |
| Rekaman terdampak | 49 dari 153 | 41 dari 44 |
| Rentetan terpanjang (epoch berurutan) | 97 | 59 |

Pada SC, separuh lebih epoch tersebut berasal dari satu rekaman (SC4762E: 148 epoch). Pada ST, hampir seluruhnya berada di tepi jendela.

**Dampak pada Ref-1.** Kalimat pada bagian *AASM Label Mapping* ("Label *Sleep stage ?* dibuang ... begitu pula dengan label *Movement time*") perlu diganti, misalnya:

> Label *Sleep stage ?* dan *Movement time* tidak dimasukkan ke dalam perhitungan *loss* dan metrik evaluasi (diberi *ignore label*). Epoch tersebut tetap dipertahankan dalam urutan sebagai konteks temporal agar kontinuitas jendela Bi-LSTM tidak terputus.

**Dampak pada Fase 1.** Berkas hasil pre-processing menyimpan label −1 pada epoch tersebut. *Unit test* jumlah epoch (SC-78: 195.469 epoch berlabel 5 kelas) tetap berlaku, karena hanya menghitung epoch berlabel 0 sampai 4.

**Dampak pada pelatihan.** *Loss* (termasuk Focal Loss) dan semua metrik harus memakai *mask* label ≠ −1.

## K2. Jumlah kanal EEG pada cabang EEG

**Status:** Ditunda sampai tahap pemodelan.

Ref-1 menyebut cabang EEG memakai Fpz-Cz saja. Pada *baseline* GBDT, menambah Pz-Oz menaikkan MF1 dari 59,6 menjadi 65,3. Fase 1 menyimpan **kedua kanal EEG**, sehingga pilihan 1 atau 2 kanal dapat diuji sebagai skenario ablasi tanpa mengulang pre-processing. Model utama mengikuti Ref-1 sampai ablasi menunjukkan sebaliknya.

## K3. Eksperimen *data leakage* pada model akhir

**Status:** Ditunda sampai tahap pemodelan.

Selisih ±7 poin (epoch-wise 81,0 vs subject-wise 73,7) saat ini berasal dari GBDT. Eksperimen perlu diulang pada model akhir. Syarat untuk Fase 1: setiap epoch menyimpan ID subjek, ID malam, dan nomor urut epoch.

## K4. Transformasi EMG pada Fase 1

**Status:** Diputuskan (5 November 2026), berdasarkan perbandingan empat kandidat pada Bagian G notebook 02.

**Keputusan.** EMG submental pada SC diproses dengan urutan berikut: *robust scaling* per rekaman (median dan IQR, dihitung setelah *trimming*), pemotongan nilai pada ±20 IQR, lalu transformasi `arcsinh`. EMG tidak disaring dan tidak di-*upsample*; cabang EMG menerima 30 nilai per epoch.

**Alasan.**

1. Nilai negatif terdapat pada 4,2% sampel EMG di dalam jendela *trimming*, tersebar pada 119 dari 153 rekaman, sehingga logaritma langsung tidak dapat dipakai. Pada kandidat logaritma dengan batas bawah 10⁻³, 4,274% nilai terbuang dan AUC pemisah Wake dan REM paling rendah (0,817).
2. Median EMG per rekaman berkisar dari −0,99 sampai 17,79 dan IQR dari 0,07 sampai 5,15, sehingga skala antar rekaman tidak sebanding dan normalisasi per rekaman diperlukan.
3. *Robust scaling* saja menyisakan distribusi yang miring (*skewness* −2,421). Menambahkan `arcsinh`, yang berperilaku seperti logaritma pada nilai besar tetapi terdefinisi untuk nilai negatif, menurunkannya menjadi −0,874 dan memberi AUC tertinggi (0,850, dibandingkan 0,842 pada data mentah).

**Catatan jujur.** Selisih AUC terhadap data mentah hanya 0,008. Alasan utama memilih `arcsinh` adalah distribusi yang lebih simetris dan terbatas bagi jaringan saraf, bukan peningkatan daya pisah yang besar.

**Dampak pada Ref-1.** Kalimat tentang transformasi `log(clip(x, ε))` pada EMG perlu diganti dengan deskripsi di atas. Rekomendasi pada `CATATAN_PERUBAHAN_REF1.md` nomor 6 dan daftar periksa pada `ANALISIS_PROJECT_SLEEP_STAGING.md` bagian 7 yang masih menyebut transformasi logaritma digantikan oleh keputusan ini.

**Catatan mutu.** Pemeriksaan pada Bagian I menandai 17 rekaman dengan EMG hampir datar atau tidak membedakan REM dari Wake (11 di antaranya subjek berusia 80 tahun ke atas). Rekaman tersebut tidak dibuang; tandanya disimpan pada `results/tables/02_cache_manifest.csv` untuk uji sensitivitas pada tahap ablasi.

## K5. Rancangan *baseline* klasik (Fase 2)

**Status:** Diputuskan (11 November 2026), diterapkan pada `notebooks/03_baseline.ipynb`.

**Keputusan.**

1. **Fitur:** 33 fitur per *epoch* yang dihitung dari *cache* Fase 1: 12 fitur untuk tiap kanal EEG (daya total, daya relatif enam pita, rasio theta/alfa dan delta/beta, parameter Hjorth, kurtosis), 5 fitur EOG, dan 4 fitur EMG. Fitur dua *epoch* sebelum dan sesudah ditempelkan (165 kolom); tetangga dari rekaman lain diisi NaN.
2. **Model:** `HistGradientBoostingClassifier` (laju belajar 0,1, maksimum 31 daun, `max_bins = 63`, bobot kelas seimbang), dengan penghentian dini pada *log-loss* subjek validasi (berhenti setelah 20 iterasi tanpa perbaikan, maksimum 500).
3. **Validasi:** SC-20 memakai satu subjek per lipatan (20 lipatan) dan SC-78 memakai 10 lipatan `GroupKFold`. Di setiap lipatan, sekitar 10% subjek latih disisihkan sebagai validasi. *Epoch* berlabel −1 tidak dilatih maupun dinilai, dan metrik dihitung dari prediksi gabungan seluruh lipatan.
4. **Cakupan analisis:** ablasi modalitas dan pemeriksaan kebocoran dijalankan pada SC-20; pentingnya fitur dan skor per subjek pada SC-78.

**Alasan.**

1. Fitur dipilih dari ciri yang dipakai dalam penilaian tahap tidur (alfa, theta, *spindle*, gelombang lambat, gerakan mata, tonus otot) dan dihitung dengan rumus pemrosesan sinyal baku. Daya relatif dipakai karena sinyal sudah dinormalisasi per rekaman sehingga daya mutlak antar-subjek kurang bermakna.
2. Skema lipatan mengikuti protokol yang lazim di literatur (SC-20 satu subjek per lipatan, SC-78 sepuluh lipatan) agar sebanding dan selaras dengan rencana Fase 3 dan 4.
3. Penghentian dini memakai *log-loss*, bukan *Macro-F1*: penghitungan *Macro-F1* di setiap iterasi mengharuskan prediksi ulang seluruh set validasi, sehingga satu konfigurasi SC-20 memerlukan lebih dari 10 menit. Untuk model *deep learning* (Fase 3), penghentian dini tetap direncanakan memakai *Macro-F1* validasi.
4. `max_bins = 63` memangkas waktu pelatihan sekitar separuh dengan *Macro-F1* yang sama atau sedikit lebih baik pada uji satu lipatan (SC-20 79,6 dan 79,6; SC-78 74,0 dibanding 73,7).

**Catatan.** Selain penghentian dini, tidak ada penyetelan hiperparameter. Pilihan ini disengaja agar *baseline* tetap sederhana dan tidak disetel berlebihan.

**Hasil.** SC-20: MF1 79,2, Kappa 0,784, F1-N1 49,3. SC-78: MF1 76,5, Kappa 0,739, F1-N1 51,0. Angka ini menjadi batas bawah Fase 3.
