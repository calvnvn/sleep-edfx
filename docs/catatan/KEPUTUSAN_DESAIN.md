# Catatan Keputusan Desain

Berkas ini mencatat keputusan rancangan yang diambil selama pengerjaan, beserta alasan dan dampaknya pada Ref-1 dan Fase 1. Tiap keputusan memiliki status: **Diputuskan**, **Ditunda**, atau **Terbuka**.

## K1. Penanganan epoch _Unscored_ dan _Movement time_ di dalam jendela _trimming_

**Status:** Diputuskan (30 Oktober 2026).

**Keputusan.** Epoch berlabel _Sleep stage ?_ (_Unscored_) dan _Movement time_ yang masih berada di dalam jendela _trimming_ **tidak dibuang dari urutan**. Sinyalnya dipertahankan, labelnya diberi nilai khusus (−1), dan epoch tersebut **dikecualikan dari perhitungan _loss_ maupun metrik** (MF1, κ, matriks konfusi).

**Alasan.**

1. Bi-LSTM membaca jendela L epoch berurutan. Jika epoch di tengah malam dibuang lalu sisanya disambung, epoch yang semula terpisah menjadi tampak bersebelahan sehingga konteks waktunya keliru.
2. Dengan nilai −1, urutan waktu tetap utuh dan model tetap dapat memakai epoch tersebut sebagai konteks bagi epoch di sekitarnya.
3. Secara efektif epoch itu tetap "dibuang" dari penilaian, sesuai maksud Ref-1.

**Dasar angka** (diukur dari hypnogram, setelah _trimming_ ±30 menit).

|                                                                     | SC-78                     | ST                        |
| ------------------------------------------------------------------- | ------------------------- | ------------------------- |
| Epoch berlabel _?_ atau _Movement time_ di dalam jendela            | 298 (±0,15% dari 195.767) | 1.690 (±3,8% dari 44.157) |
| Di tengah malam (di antara tidur pertama dan terakhir)              | 287                       | 82                        |
| Di tepi jendela (sebelum tidur pertama atau sesudah tidur terakhir) | 11                        | 1.608                     |
| Rekaman terdampak                                                   | 49 dari 153               | 41 dari 44                |
| Rentetan terpanjang (epoch berurutan)                               | 97                        | 59                        |

Pada SC, separuh lebih epoch tersebut berasal dari satu rekaman (SC4762E: 148 epoch). Pada ST, hampir seluruhnya berada di tepi jendela.

**Dampak pada Ref-1.** Kalimat pada bagian _AASM Label Mapping_ ("Label _Sleep stage ?_ dibuang ... begitu pula dengan label _Movement time_") perlu diganti, misalnya:

> Label _Sleep stage ?_ dan _Movement time_ tidak dimasukkan ke dalam perhitungan _loss_ dan metrik evaluasi (diberi _ignore label_). Epoch tersebut tetap dipertahankan dalam urutan sebagai konteks temporal agar kontinuitas jendela Bi-LSTM tidak terputus.

**Dampak pada Fase 1.** Berkas hasil pre-processing menyimpan label −1 pada epoch tersebut. _Unit test_ jumlah epoch (SC-78: 195.469 epoch berlabel 5 kelas) tetap berlaku, karena hanya menghitung epoch berlabel 0 sampai 4.

**Dampak pada pelatihan.** _Loss_ (termasuk Focal Loss) dan semua metrik harus memakai _mask_ label ≠ −1.

## K2. Jumlah kanal EEG pada cabang EEG

**Status:** Ditunda sampai tahap pemodelan.

Ref-1 menyebut cabang EEG memakai Fpz-Cz saja. Pada _baseline_ GBDT, menambah Pz-Oz menaikkan MF1 dari 59,6 menjadi 65,3. Fase 1 menyimpan **kedua kanal EEG**, sehingga pilihan 1 atau 2 kanal dapat diuji sebagai skenario ablasi tanpa mengulang pre-processing. Model utama mengikuti Ref-1 sampai ablasi menunjukkan sebaliknya.

## K3. Eksperimen _data leakage_ pada model akhir

**Status:** Ditunda sampai tahap pemodelan.

Selisih ±7 poin (epoch-wise 81,0 vs subject-wise 73,7) saat ini berasal dari GBDT. Eksperimen perlu diulang pada model akhir. Syarat untuk Fase 1: setiap epoch menyimpan ID subjek, ID malam, dan nomor urut epoch.

## K4. Transformasi EMG pada Fase 1

**Status:** Diputuskan (5 November 2026), berdasarkan perbandingan empat kandidat pada Bagian G notebook 02.

**Keputusan.** EMG submental pada SC diproses dengan urutan berikut: _robust scaling_ per rekaman (median dan IQR, dihitung setelah _trimming_), pemotongan nilai pada ±20 IQR, lalu transformasi `arcsinh`. EMG tidak disaring dan tidak di-_upsample_; cabang EMG menerima 30 nilai per epoch.

**Alasan.**

1. Nilai negatif terdapat pada 4,2% sampel EMG di dalam jendela _trimming_, tersebar pada 119 dari 153 rekaman, sehingga logaritma langsung tidak dapat dipakai. Pada kandidat logaritma dengan batas bawah 10⁻³, 4,274% nilai terbuang dan AUC pemisah Wake dan REM paling rendah (0,817).
2. Median EMG per rekaman berkisar dari −0,99 sampai 17,79 dan IQR dari 0,07 sampai 5,15, sehingga skala antar rekaman tidak sebanding dan normalisasi per rekaman diperlukan.
3. _Robust scaling_ saja menyisakan distribusi yang miring (_skewness_ −2,421). Menambahkan `arcsinh`, yang berperilaku seperti logaritma pada nilai besar tetapi terdefinisi untuk nilai negatif, menurunkannya menjadi −0,874 dan memberi AUC tertinggi (0,850, dibandingkan 0,842 pada data mentah).

**Catatan jujur.** Selisih AUC terhadap data mentah hanya 0,008. Alasan utama memilih `arcsinh` adalah distribusi yang lebih simetris dan terbatas bagi jaringan saraf, bukan peningkatan daya pisah yang besar.

**Dampak pada Ref-1.** Kalimat tentang transformasi `log(clip(x, ε))` pada EMG perlu diganti dengan deskripsi di atas. Rekomendasi pada `CATATAN_PERUBAHAN_REF1.md` nomor 6 dan daftar periksa pada `ANALISIS_PROJECT_SLEEP_STAGING.md` bagian 7 yang masih menyebut transformasi logaritma digantikan oleh keputusan ini.

**Catatan mutu.** Pemeriksaan pada Bagian I menandai 17 rekaman dengan EMG hampir datar atau tidak membedakan REM dari Wake (11 di antaranya subjek berusia 80 tahun ke atas). Rekaman tersebut tidak dibuang; tandanya disimpan pada `results/tables/02_cache_manifest.csv` untuk uji sensitivitas pada tahap ablasi.

## K5. Rancangan _baseline_ klasik (Fase 2)

**Status:** Diputuskan (11 November 2026), diterapkan pada `notebooks/03_baseline.ipynb`.

**Keputusan.**

1. **Fitur:** 33 fitur per _epoch_ yang dihitung dari _cache_ Fase 1: 12 fitur untuk tiap kanal EEG (daya total, daya relatif enam pita, rasio theta/alfa dan delta/beta, parameter Hjorth, kurtosis), 5 fitur EOG, dan 4 fitur EMG. Fitur dua _epoch_ sebelum dan sesudah ditempelkan (165 kolom); tetangga dari rekaman lain diisi NaN.
2. **Model:** `HistGradientBoostingClassifier` (laju belajar 0,1, maksimum 31 daun, `max_bins = 63`, bobot kelas seimbang), dengan penghentian dini pada _log-loss_ subjek validasi (berhenti setelah 20 iterasi tanpa perbaikan, maksimum 500).
3. **Validasi:** SC-20 memakai satu subjek per lipatan (20 lipatan) dan SC-78 memakai 10 lipatan `GroupKFold`. Di setiap lipatan, sekitar 10% subjek latih disisihkan sebagai validasi. _Epoch_ berlabel −1 tidak dilatih maupun dinilai, dan metrik dihitung dari prediksi gabungan seluruh lipatan.
4. **Cakupan analisis:** ablasi modalitas dan pemeriksaan kebocoran dijalankan pada SC-20; pentingnya fitur dan skor per subjek pada SC-78.

**Alasan.**

1. Fitur dipilih dari ciri yang dipakai dalam penilaian tahap tidur (alfa, theta, _spindle_, gelombang lambat, gerakan mata, tonus otot) dan dihitung dengan rumus pemrosesan sinyal baku. Daya relatif dipakai karena sinyal sudah dinormalisasi per rekaman sehingga daya mutlak antar-subjek kurang bermakna.
2. Skema lipatan mengikuti protokol yang lazim di literatur (SC-20 satu subjek per lipatan, SC-78 sepuluh lipatan) agar sebanding dan selaras dengan rencana Fase 3 dan 4.
3. Penghentian dini memakai _log-loss_, bukan _Macro-F1_: penghitungan _Macro-F1_ di setiap iterasi mengharuskan prediksi ulang seluruh set validasi, sehingga satu konfigurasi SC-20 memerlukan lebih dari 10 menit. Untuk model _deep learning_ (Fase 3), penghentian dini tetap direncanakan memakai _Macro-F1_ validasi.
4. `max_bins = 63` memangkas waktu pelatihan sekitar separuh dengan _Macro-F1_ yang sama atau sedikit lebih baik pada uji satu lipatan (SC-20 79,6 dan 79,6; SC-78 74,0 dibanding 73,7).

**Catatan.** Selain penghentian dini, tidak ada penyetelan hiperparameter. Pilihan ini disengaja agar _baseline_ tetap sederhana dan tidak disetel berlebihan.

**Hasil.** SC-20: MF1 79,2, Kappa 0,784, F1-N1 49,3. SC-78: MF1 76,5, Kappa 0,739, F1-N1 51,0. Angka ini menjadi batas bawah Fase 3.

## K6. Rancangan model _deep learning_ awal pada SC-20 (Fase 3)

**Status:** Diputuskan (15 November 2026), diterapkan pada `notebooks/04_cnn_model.ipynb`.

**Keputusan.**

1. **Arsitektur:** _Multi-Branch 1D-CNN + Bi-LSTM_ dengan tiga cabang ekstraktor fitur terpisah sesuai laju pencuplikan tiap modalitas: cabang EEG (2 kanal, 100 Hz; kernel awal 50 dengan _stride_ 6), cabang EOG (1 kanal, 100 Hz; arsitektur identik dengan EEG), dan cabang EMG (1 kanal, 1 Hz, 30 titik per _epoch_; kernel kecil 5–3 tanpa _downsampling_ agresif). Keluaran ketiga cabang digabungkan (_concatenation_ 256 dimensi) lalu diproses oleh Bi-LSTM 2 lapis (`hidden_size = 128` per arah, _dropout_ 0,3) pada jendela berurutan sepanjang $L = 20$ _epoch_ (10 menit).
2. **Fungsi kerugian dan optimasi:** _Weighted Focal Loss_ ($\gamma = 2,0$) dengan bobot kelas _inverse square-root_ yang dinormalisasi serta `ignore_index = -1` agar _epoch_ _Unscored_ dan _Movement time_ tetap menjaga kontinuitas urutan Bi-LSTM tanpa dihitung dalam _loss_. Pelatihan memakai optimizer AdamW (`lr = 1e-3`, `weight_decay = 1e-4`), penjadwal `CosineAnnealingLR`, pemotongan gradien (`max_norm = 1,0`), serta _Automatic Mixed Precision_ (AMP) saat berjalan di GPU CUDA.
3. **Validasi dan penghentian dini:** Evaluasi mengikuti skema 20 lipatan satu subjek per lipatan (_Leave-One-Subject-Out_) pada SC-20 (39 rekaman, subjek `00`–`19`). Di setiap lipatan, satu subjek terpisah disisihkan sebagai set validasi untuk _early stopping_ berbasis **Macro-F1 validasi** (berhenti setelah 6 _epoch_ tanpa perbaikan, maksimum 25 _epoch_), sesuai rencana pada K5. Saat inferensi validasi dan uji, rekaman dipotong tanpa tumpang-tindih (`stride = 20`) agar tiap _epoch_ dinilai tepat satu kali.
4. **Penyesuaian perangkat keras tim:** Konfigurasi pelatihan mendeteksi perangkat secara otomatis (`USE_CUDA = DEVICE.type == "cuda"`): berjalan penuh 20 lipatan pada GPU NVIDIA dan beralih ke mode pemeriksaan cepat 2 lipatan pada mesin khusus CPU agar seluruh anggota tim dapat menjalankan buku catatan tanpa hambatan perangkat keras.

**Alasan.**

1. Pemisahan cabang CNN mencegah laju pencuplikan EMG (1 Hz) terdistorsi oleh _downsampling_ besar yang dibutuhkan sinyal EEG/EOG (100 Hz), sekaligus memungkinkan masing-masing cabang mempelajari filter temporal khas tiap modalitas.
2. Sesuai catatan pada K5, penghentian dini pada Fase 3 memakai _Macro-F1_ validasi (bukan _log-loss_) agar bobot terbaik yang disimpan langsung mengoptimalkan metrik utama proyek dan sensitif terhadap kelas minoritas N1.
3. Pemisahan subjek validasi dari subjek uji memastikan tidak ada kebocoran data (_data leakage_) selama pemilihan _checkpoint_ terbaik maupun penyetelan ambang penghentian dini.

**Dampak pada Ref-1 dan K2/K3.**

1. **Keterkaitan dengan K2:** Model dasar Fase 3 memakai 2 kanal EEG (`Fpz-Cz` dan `Pz-Oz`) sebagai patokan awal arsitektur multimodal; perbandingan 1 kanal versus 2 kanal dijadwalkan pada tahap ablasi.
2. **Keterkaitan dengan K3:** Seluruh pelatihan pada `04_cnn_model.ipynb` menerapkan pemisahan berbasis ID subjek (_subject-wise_), dan hasil SC-20 dibandingkan langsung dengan batas bawah _baseline_ klasik (MF1 79,2; Kappa 0,784; F1-N1 49,3) pada `results/tables/04_sc20_comparison.csv`.

**Hasil.** SC-20: ACC 82,7, MF1 78,6, Kappa 0,768, F1-N1 50,2 (unggul pada F1-N1 dibanding baseline 49,3; sedikit di bawah MF1 baseline 79,2 karena ukuran data latih SC-20 masih 18 subjek per lipatan sebelum skala penuh SC-78 pada Fase 4).
