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

## K6. Rancangan perombakan Fase 3 pada SC-20 (CNN multi-cabang + Bi-LSTM)

**Status:** Diputuskan (16 November 2026), diterapkan pada `notebooks/04_cnn_bilstm_sc20.ipynb` (cabang `feat/phase3-rework`). Hanya SC-20 yang dijalankan; SC-78 dan ST belum.

**Keputusan.**

1. **Model:** tiga cabang CNN 1D per *epoch* (EEG 2 kanal dan EOG dengan dua jalur kernel paralel 50 dan 400; EMG dengan dua konvolusi kecil) menjadi vektor 224 angka, lalu Bi-LSTM dua lapis (128 unit per arah) dan kepala linear 5 kelas per *epoch* (*many-to-many*). Total 1.089.477 parameter.
2. **Jendela:** 20 *epoch* (10 menit). Geser 10 *epoch* saat pelatihan dan saat inferensi, dengan peluang jendela tumpang tindih dirata-ratakan. Jendela tidak melintasi rekaman; label −1 tetap berada di urutan tetapi tidak dihitung pada *loss*.
3. ***Loss*:** *cross-entropy* (CE) sebagai kontrol dan *focal loss* ($\gamma$ = 2) yang dibenahi: $p_t$ adalah peluang kelas benar dan bobot kelas hanya pengali. Bobot kelas berbanding terbalik dengan akar frekuensi, dinormalkan rerata 1 (SC-20: Wake 0,89, N1 1,52, N2 0,60, N3 1,07, REM 0,92).
4. **Validasi:** SC-20 satu subjek per lipatan; sekitar 10% subjek latih (2 subjek) sebagai validasi, dipilih identik dengan notebook 03 (diperiksa dengan `assert`). Penghentian dini pada Macro-F1 validasi (kesabaran 6, maksimum 30 putaran). Pemilihan *loss* memakai Macro-F1 validasi, bukan nilai uji.
5. **Pengulangan:** tiga *seed* (0, 1, 2) per *loss*, dilaporkan rerata ± simpangan baku, dengan uji Wilcoxon berpasangan per subjek (n = 20).
6. **Pengujian sebelum pelatihan:** bentuk keluaran, uji tiga sifat *focal loss*, kesamaan subjek validasi dengan notebook 03, dan uji menghafal 16 jendela (akurasi 100%).

**Alasan.**

1. Versi pertama Fase 3 (cabang rekan tim) memiliki *focal loss* yang salah rumus ($p_t = p^w$), tanpa pembanding CE, validasi satu subjek, dan tanpa pengulangan; selisih kecil terhadap *baseline* sehingga tidak dapat ditafsirkan. Perombakan menutup celah tersebut satu per satu.
2. Panjang jendela 20 *epoch* dipilih sebagai kompromi konteks dan memori GPU 6 GB; belum dibandingkan dengan panjang lain.
3. Subjek validasi yang sama dengan *baseline* menghilangkan perbedaan pembagian data sebagai penjelas selisih hasil.

**Hasil (SC-20, rerata ± simpangan baku tiga *seed*).**

| | ACC | MF1 | $\kappa$ | F1-N1 |
|---|---|---|---|---|
| CE | 85,1 ± 0,1 | 80,1 ± 0,4 | 0,795 ± 0,001 | 53,0 ± 1,1 |
| *Focal* | 84,5 ± 0,6 | 80,2 ± 0,9 | 0,790 ± 0,008 | 53,4 ± 2,4 |
| *Baseline* (K5) | 84,1 | 79,2 | 0,784 | 49,3 |

Uji berpasangan per subjek: CE vs *baseline* selisih +0,64 poin (p = 0,475), *focal* vs *baseline* +0,56 (p = 0,202), *focal* vs CE −0,07 (p = 0,927). Tidak ada yang signifikan.

**Catatan jujur.**

1. **Keunggulan sekitar satu poin MF1 atas *baseline* belum terbukti**; yang lebih jelas hanya F1-N1 (sekitar +4 poin) dan ACC pada CE.
2. CE terpilih oleh Macro-F1 validasi (82,5 vs 82,4), tetapi selisih 0,1 poin adalah seri praktis. CE dipertahankan karena lebih sederhana dan lebih stabil antar-*seed*.
3. CE dan *focal* menukar profil: CE lebih baik pada Wake, N2, dan REM, *focal* pada N1 (sensitivitas 60,1% vs 50,5%) dan N3.
4. Penurunan Wake dan N2 pada versi pertama **tidak terulang** (sensitivitas Wake CE 89,4 dan *focal* 87,9 vs *baseline* 87,5; N2 88,1 dan 84,9 vs 85,0). Penyebabnya tidak dapat dipisahkan karena *loss*, validasi, dan arsitektur diubah sekaligus; dugaan galat *focal loss* konsisten dengan hasil tetapi belum diuji tersendiri.
5. Hanya satu cabang Fase 3 yang akan digabung ke cabang utama; pemilihannya dibicarakan dengan rekan tim.
