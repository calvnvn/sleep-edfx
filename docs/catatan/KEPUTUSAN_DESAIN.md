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
