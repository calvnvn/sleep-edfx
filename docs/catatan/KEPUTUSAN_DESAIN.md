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

**Status:** Diputuskan (22 November 2026) berdasarkan ablasi Fase 5 (K8).

**Keputusan.** Model tetap memakai **dua kanal EEG**.

**Bukti.** Pada model CNN + Bi-LSTM, menambah Pz-Oz ke Fpz-Cz menaikkan MF1 per subjek +0,57 poin (interval 95% −0,34 sampai +1,51), membaik pada 12 dari 20 subjek, p = 0,26. Kanal kedua belum terbukti membantu dan juga tidak terbukti merugikan. Dua kanal dipertahankan karena sudah dipakai pada model referensi dan benchmark 78 subjek (K7).

**Rancangan awal.** Ref-1 menyebut cabang EEG memakai Fpz-Cz saja. Pada *baseline* GBDT, menambah Pz-Oz menaikkan MF1 dari 59,6 menjadi 65,3. Fase 1 menyimpan **kedua kanal EEG**, sehingga pilihan 1 atau 2 kanal dapat diuji sebagai skenario ablasi tanpa mengulang pre-processing. Model utama mengikuti Ref-1 sampai ablasi menunjukkan sebaliknya.

## K3. Eksperimen *data leakage* pada model akhir

**Status:** Selesai (22 November 2026) pada ablasi Fase 5 (K8).

**Hasil.** Pada model CNN + Bi-LSTM (satu *seed*), pembagian acak per *epoch* menaikkan MF1 dari 80,4 ke 90,6 (+10,2 poin), akurasi dari 85,1 ke 93,3, F1-N1 dari 54,2 ke 75,4, dan κ dari 0,796 ke 0,908. Pada *baseline* GBDT (notebook 03), MF1 naik dari 79,2 ke 86,2 (+7,0). Model yang lebih besar terkena lebih parah; penyebabnya masih dugaan (menghafal ciri khas satu malam rekaman). Skor pada pembagian acak tidak dilaporkan sebagai hasil model.

**Rancangan awal.** Selisih ±7 poin (epoch-wise 81,0 vs subject-wise 73,7) saat ini berasal dari GBDT. Eksperimen perlu diulang pada model akhir. Syarat untuk Fase 1: setiap epoch menyimpan ID subjek, ID malam, dan nomor urut epoch.

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
3. ***Loss*:** *cross-entropy* (CE) **tanpa bobot kelas** sebagai kontrol dan *focal loss* ($\gamma$ = 2) yang dibenahi **dengan bobot kelas**: $p_t$ adalah peluang kelas benar dan bobot kelas hanya pengali. Bobot kelas berbanding terbalik dengan akar frekuensi, dinormalkan rerata 1 (SC-20: Wake 0,89, N1 1,52, N2 0,60, N3 1,07, REM 0,92). Lihat koreksi di akhir bagian ini.
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

**Koreksi (19 November 2026).** Pembandingan "CE vs *focal*" di atas mengubah **dua** hal sekaligus: CE dijalankan tanpa bobot kelas, sedangkan *focal* memakai bobot kelas. Akibatnya:

- Uji pertama *focal loss* di notebook 04 membandingkan $\gamma = 0$ **tanpa bobot** dengan CE tanpa bobot, bukan dengan "CE berbobot" seperti yang tertulis di dokumen penjelasan dan *markdown* notebook. Kodenya benar; uraiannya yang keliru.
- Perbedaan profil pada catatan nomor 3 (N1 dan N3 lebih baik pada *focal*) tidak bisa diatribusikan ke *focal loss*. Bobot kelas sendiri juga bisa menjelaskannya.

Kedua faktor dipisahkan pada Fase 5 dengan desain 2×2 (keputusan K8).

## K7. Benchmark SC-78 dengan konfigurasi beku (Fase 4)

**Status:** Diputuskan (19 November 2026), diterapkan pada `notebooks/05_cnn_bilstm_sc78.ipynb`.

**Keputusan.**

1. **Konfigurasi dibekukan** sama persis dengan K6: CE tanpa bobot kelas, jendela 20 *epoch*, hiperparameter notebook 04. Tidak ada yang disetel pada SC-78.
2. **Fold identik dengan notebook 03:** `GroupKFold(10)` atas *epoch* berlabel yang dikelompokkan per subjek, lalu sekitar 10% subjek latih sebagai validasi (`GroupShuffleSplit`, `random_state = 0 + fold`). Bukti: MF1 per *fold* baseline yang dihitung ulang pada *fold* ini sama dengan nilai tersimpan (76,4 ± 2,4).
3. **Tiga *seed*** (0, 1, 2), dilaporkan rerata ± simpangan baku.
4. **Pembanding utama:** uji berpasangan per subjek (n = 78) terhadap baseline, dengan interval *bootstrap* 95% dan Wilcoxon. Analisis tambahan: kelompok usia, rekaman EMG bermasalah, dan subjek SC-20 yang dilatih dengan 17 vs sekitar 63 subjek.
5. **Kode bersama** di `src/sleep_dl.py`, yang dibuktikan setara dengan notebook 04 oleh `tests/test_sleep_dl.py` (bobot awal, keluaran, dan satu pelatihan pendek di CPU identik bit demi bit).
6. **Data di-*memory-map***, tidak disalin ke RAM, karena SC-78 butuh sekitar 3,5 GB untuk EEG/EOG.

**Alasan.**

1. SC-20 adalah bagian dari SC-78. Bila konfigurasi dipilih ulang memakai hasil uji SC-20 (misalnya dari ablasi Fase 5), angka SC-78 tidak lagi bersih. Membekukan konfigurasi sebelum menjalankan SC-78 mencegah hal itu. Data yang benar-benar independen hanya ST (Fase 6).
2. Fold yang sama membuat perbandingan per subjek dengan baseline sah. Tanpa itu, selisih bisa berasal dari pembagian data.
3. Baseline GBDT praktis deterministik (validasi diberikan eksplisit, tanpa *subsampling*), sehingga variasi antar-*run* hanya ada pada model DL. Karena itu DL diulang tiga kali, baseline cukup satu.
4. Kode yang disalin ke beberapa notebook adalah sumber ketidaksesuaian pada versi pertama Fase 3. Satu modul yang diuji menghilangkan risiko itu.

**Hasil (20 November 2026; SC-78, rerata ± simpangan baku tiga *seed*, sekitar 2,8 jam GPU).**

| | ACC | MF1 | $\kappa$ | F1-N1 | F1-N3 |
|---|---|---|---|---|---|
| CNN + Bi-LSTM (CE) | 83,6 ± 0,4 | 78,0 ± 0,6 | 0,774 ± 0,005 | 52,7 ± 0,8 | 73,7 ± 1,8 |
| *Baseline* (K5) | 80,7 | 76,5 | 0,739 | 51,0 | 74,6 |

Per subjek (n = 78): rerata selisih +1,06 poin, interval *bootstrap* 95% [−0,26; +2,29]; median +1,6; CNN lebih baik pada 48 subjek; Wilcoxon p = 0,0052. Simpangan baku MF1 antar-*fold*: CNN 3,1, *baseline* 2,4.

**Catatan jujur.**

1. **CNN lebih baik untuk subjek tipikal, tetapi rerata keunggulannya per subjek belum terbukti.** Tiga subjek turun jauh di bawah *baseline*: subjek 36 (−25,0; REM tidak terdeteksi, N3 → N2), subjek 17 (−15,4; Wake → REM dan N1), dan subjek 25 (−15,3; N3 tidak pernah diprediksi). Kegagalan ini muncul pada ketiga *seed*. Penyebabnya belum diselidiki.
2. **Perbedaan sensitivitas N1 (−9,3) dan N3 (−11,4) terutama berasal dari bobot kelas, bukan dari arsitektur.** *Baseline* berbobot seimbang dan menebak N1 pada 14,9% *epoch* (porsi sebenarnya 11,0%) dan N3 pada 8,5% (6,7%). Porsi tebakan CNN hampir sama dengan porsi sebenarnya (10,2% dan 6,6%), dan *precision* N1 serta N3-nya lebih tinggi (54,8 vs 44,3; 74,2 vs 66,7).
3. Keunggulan tidak berkorelasi dengan usia (r = 0,07). Kedua model kehilangan sekitar 10 poin MF1 dari kelompok 25–39 ke 80+. Uji per kelompok usia bersifat eksploratif; setelah koreksi Holm hanya kelompok 25–39 yang di bawah 0,05, dan kelompok itu adalah subjek SC-20 yang dipakai selama pengembangan.
4. Rekaman dengan EMG bermasalah menurunkan F1-REM kedua model, tetapi tidak merugikan CNN lebih dari *baseline* (subjek 60+ bertanda: +3,3; tanpa tanda: +1,1). Rekaman tersebut tetap dipakai.
5. Untuk 20 subjek SC-20, pelatihan dengan sekitar 63 subjek lebih baik daripada dengan 17 subjek (+1,15 poin, 14 dari 20 subjek, p = 0,030). Jumlah dan keragaman data latih berubah bersamaan, sehingga tidak dapat dipisahkan.
6. *Early stopping* memilih putaran ke-5 (median, rentang 1–18), dan MF1 validasi berfluktuasi beberapa poin antar-putaran. Pemilihan titik henti dengan 7–8 subjek validasi berisik. Perbaikannya (misalnya rerata bobot beberapa putaran) sengaja tidak dikerjakan agar konfigurasi tetap beku.
7. *Run* sempat dihentikan sistem karena RAM laptop hampir habis, lalu dilanjutkan dari *fold* tersimpan. Satu galat ditemukan dan diperbaiki: pada Windows, `run_cv` tidak menutup berkas *partial* sehingga berkas itu gagal dihapus. Hasil tidak terpengaruh.

## K8. Rancangan ablasi pada SC-20 (Fase 5)

**Status:** Diputuskan (19 November 2026), diterapkan pada `notebooks/06_ablasi_sc20.ipynb`.

**Keputusan.**

1. **Referensi:** model K6 dengan CE tanpa bobot. Setiap lengan mengubah **satu** hal terhadap referensi.
2. **Lengan dan pertanyaannya:**

| Pertanyaan | Lengan | Yang berubah |
|---|---|---|
| *Loss* (2×2) | `ce`, `ce_w`, `focal_uw`, `focal` | γ ∈ {0, 2} × bobot kelas {tidak, ya} |
| Konteks waktu | `cnn_only` | Bi-LSTM dibuang; tiap *epoch* diklasifikasi dari fitur CNN-nya sendiri |
| Panjang jendela | `seq10`, `seq30` | L = 10 dan 30; jumlah *epoch* per *batch* dijaga sekitar 640 |
| Modalitas (kumulatif) | `eeg_fpz`, `eeg_2ch`, `eeg_eog`, `ce` | Fpz-Cz → + Pz-Oz (K2) → + EOG → + EMG |
| Kebocoran data (K3) | `leak` | Split acak per *epoch* seperti notebook 03; 1 *seed* |

3. **Lengan `ce` dan `focal` dipakai ulang dari notebook 04.** Konfigurasinya diperiksa sama persis, dan modul dibuktikan setara dengan kode notebook 04.
4. **Bobot kelas pada `ce_w`** dihitung sebagai rerata suku CE berbobot (bukan rerata berbobot bawaan PyTorch), sehingga `ce_w` sama persis dengan *focal* pada γ = 0. Dengan begitu keempat sel 2×2 berbeda hanya pada γ dan bobot.
5. **Statistik:** selisih berpasangan per subjek terhadap referensi (rerata 3 *seed*), interval *bootstrap* 95%, Wilcoxon dengan koreksi Holm untuk 9 perbandingan; efek utama dan interaksi pada 2×2.
6. **Tidak dikerjakan:** re-implementasi TinySleepNet (E10 pada dokumen analisis) dan γ = 1.

**Alasan.**

1. Pada Fase 3, CE tanpa bobot dibandingkan dengan *focal* berbobot, sehingga dua faktor berubah bersamaan (lihat koreksi K6). Desain 2×2 memisahkan keduanya.
2. Tangga modalitas sengaja sama dengan tangga baseline (notebook 03, H1), agar kontribusi tiap modalitas bisa dibandingkan antara GBDT dan DL.
3. Lengan `cnn_only` mempertahankan jendela dan *batch* yang sama sehingga hanya Bi-LSTM yang hilang; perbedaan jumlah *update* dan *batch* tidak ikut memengaruhi.
4. Menjaga jumlah *epoch* per *batch* pada L = 10 dan 30 membuat jumlah langkah optimasi per putaran latih tetap sama, sehingga yang berubah hanya panjang konteks.
5. Kebocoran memakai satu *seed* karena efek yang dicari besar (7 poin pada baseline), jauh di atas variasi antar-*seed* (simpangan baku sekitar 0,4 poin).
6. Semua keputusan pemilihan memakai Macro-F1 validasi; data uji hanya dilaporkan. Subjek uji SC-20 dipakai oleh semua lengan, jadi ablasi menjelaskan kontribusi komponen, bukan memilih model untuk SC-78 (K7).
7. TinySleepNet mahal dibuat setia pada aslinya, dan pertanyaan inti proyek (modalitas, *loss*, konteks) sudah dijawab oleh lengan di atas. γ = 1 tidak menambah pertanyaan baru selama γ = 2 pun belum terbukti berbeda dari γ = 0.

**Hasil (22 November 2026; SC-20, 20 subjek berusia 25–34 tahun, rerata tiga *seed*, sekitar 11,9 jam GPU untuk lengan baru dan 1,7 jam untuk lengan kebocoran).**

| Lengan | MF1 | κ | Selisih per subjek terhadap referensi (interval 95%) | *p* (Holm, 9 perbandingan) |
|---|---|---|---|---|
| `ce` (referensi) | 80,1 ± 0,4 | 0,795 | | |
| `focal` | 80,2 ± 0,9 | 0,790 | −0,07 (−1,13; +0,95) | 1,00 |
| `ce_w` | 80,7 ± 0,3 | 0,795 | +0,39 (−0,60; +1,36) | 1,00 |
| `focal_uw` | 79,9 ± 0,2 | 0,791 | −0,42 (−1,42; +0,61) | 1,00 |
| `cnn_only` | 77,6 ± 0,1 | 0,769 | **−2,34 (−3,43; −1,37)** | **0,001** |
| `seq10` | 80,1 ± 0,3 | 0,793 | +0,01 (−0,73; +0,79) | 1,00 |
| `seq30` | 80,1 ± 0,8 | 0,794 | −0,03 (−0,76; +0,73) | 1,00 |
| `eeg_fpz` | 78,3 ± 0,4 | 0,778 | −1,75 (−3,21; −0,25) | 0,35 |
| `eeg_2ch` | 79,0 ± 0,3 | 0,781 | −1,19 (−2,36; +0,07) | 0,49 |
| `eeg_eog` | 79,9 ± 0,1 | 0,788 | −0,11 (−0,93; +0,77) | 1,00 |

*Baseline* pada *epoch* yang sama: MF1 79,2. Desain 2×2 (selisih per subjek, *p* tanpa koreksi): bobot kelas +0,37 (−0,45; +1,26; p = 0,76), γ = 2 −0,44 (−1,02; +0,13; p = 0,23), interaksi −0,04 (−1,28; +1,07; p = 0,87).

**Catatan jujur.**

1. **Hanya konteks waktu yang terbukti.** Tanpa Bi-LSTM, MF1 turun 2,3 poin per subjek dan 17 dari 20 subjek memburuk, terutama pada REM (−3,4 pada jendela 10), N1 (−6,6), dan Wake (−2,5). Jendela 10, 20, dan 30 *epoch* memberi MF1 yang sama. Tanpa konteks, prediksi berganti tahap 19,5 kali per jam, pakar 12,3, *baseline* 15,1, dan model dengan konteks 10,6–11,5.
2. **Bobot kelas menggeser keseimbangan, bukan menaikkan mutu.** Sensitivitas N1 naik dari sekitar 50% ke 60% dan N3 dari sekitar 81% ke 85%, tetapi *precision* N1 turun dari sekitar 56% ke 48–50% dan sensitivitas N2 turun 2–3 poin. *Focal loss* tidak menambah apa pun di atas bobot kelas. Koreksi untuk K6: pergeseran profil N1/N3 di notebook 04 berasal dari bobot kelas, bukan dari *focal loss*. CE tanpa bobot tetap dipertahankan sebagai referensi, karena tidak ada lengan yang terbukti lebih baik.
3. **Tangga modalitas jauh lebih landai pada CNN + Bi-LSTM daripada pada GBDT.** CNN: 78,3 → 79,0 → 79,9 → 80,1; GBDT tanpa konteks: 68,4 → 72,5 → 75,1 → 76,6. Perbandingan ini tidak sepadan, karena GBDT di tabel tidak memakai konteks. Tanpa konteks dan dengan semua sinyal, CNN (77,6) hanya sekitar 1 poin di atas GBDT (76,6), tanpa uji. EMG menaikkan F1-REM 1,8 poin (85,4 → 87,2) tetapi tidak MF1 keseluruhan.
4. **Daya uji rendah.** Dengan 20 subjek, lebar interval sekitar ±1 poin. "Tidak signifikan" berarti "tidak terbukti", bukan "tidak berpengaruh". Subjek hanya berusia 25–34 tahun.
5. **Lengan konfirmasi pada SC-78 tidak dijalankan** (biaya sekitar dua kali lipat; konfigurasi SC-78 sudah dibekukan lebih dulu). Temuan ini hanya berlaku untuk 20 subjek tersebut.
6. **Tiga dugaan belum diuji:** penyebab CNN lebih tahan terhadap hilangnya sinyal, penyebab kebocoran lebih parah pada model besar, dan kecenderungan model dengan konteks terlalu mulus.
7. Proses sempat terhenti beberapa kali (RAM laptop menipis, laptop masuk mode tidur) dan dilanjutkan dari *fold* tersimpan. Semua lengan selesai tanpa galat.
