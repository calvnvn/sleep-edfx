# Panduan Penulisan *Markdown* Notebook

Panduan ini mengatur teks penjelas (sel *markdown*) di notebook 01–06. Tujuannya agar notebook dapat dibaca oleh orang yang tidak ikut mengerjakan proyek, misalnya dosen penguji, rekan sekelas, atau pembaca repositori, tanpa harus membuka dokumen lain.

## 1. Dasar dari literatur

| Sumber | Temuan yang dipakai |
|---|---|
| Rule dkk. (2019), *Ten Simple Rules for Writing and Sharing Computational Analyses in Jupyter Notebooks*, PLOS Comput. Biol. 15(7) | Notebook harus bercerita untuk pembaca tertentu (aturan 1), mencatat alasan dan bukan hanya hasil (aturan 2), dan setiap sel mengerjakan satu langkah yang diberi judul (aturan 3). |
| Rule, Tabard, dan Hollan (2018), *Exploration and Explanation in Computational Notebooks*, CHI 2018 | Dari sekitar 200 notebook akademik, sebagian besar hanya menjelaskan metode; hanya sedikit yang menjelaskan alasan atau menafsirkan hasil. |
| Wang dkk. (2022), *Documentation Matters*, ACM TOCHI 29(2) | Pada 80 notebook Kaggle terbaik, penjelasan proses selalu diletakkan **sebelum** sel kode dan penafsiran hasil **sesudahnya**. Notebook yang baik juga dibuka dengan gambaran masalah dan ditutup dengan ringkasan. |

## 2. Pembaca sasaran

Pembaca diasumsikan memahami dasar *machine learning* (data latih/uji, *overfitting*, CNN), tetapi **tidak** tahu:
- isi notebook lain;
- kode keputusan internal (K1–K8);
- nama berkas;
- nama variabel atau nama percobaan di kode.

## 3. Struktur

1. **Sel pembuka.** Berisi:
   - pertanyaan penelitian dalam satu kalimat dan alasan pentingnya;
   - data yang dipakai;
   - hasil utama (ditulis di depan supaya pembaca yang berhenti di sini tetap tahu jawabannya);
   - daftar istilah yang dipakai di notebook itu;
   - peta bagian.
2. **Sebelum sel kode:** judul bagian, lalu **Tujuan** (apa yang dikerjakan dan mengapa, 1–3 kalimat) dan **Input** (data atau hasil yang masuk, dalam bahasa biasa).
3. **Sesudah sel kode:**
   - **Output:** apa yang terlihat, dengan angka;
   - **Kesimpulan:** arti angka itu dan batasannya.
4. **Sel penutup.** Berisi:
   - jawaban atas pertanyaan pembuka;
   - keterbatasan;
   - catatan teknis. Hanya di sini nama berkas, rujukan keputusan desain (K1–K8), dan detail operasional (misalnya *run* yang terputus) boleh ditulis.

## 4. Aturan bahasa

1. **Istilah teknis** didefinisikan saat pertama muncul, cukup sekali, lalu dipakai konsisten. Istilah bahasa Inggris dicetak miring.
2. **Kata *epoch*** hanya berarti potongan sinyal 30 detik. Satu kali melewati seluruh data latih disebut **putaran latih**.
3. **Notebook lain** disebut berdasarkan isinya, misalnya "*baseline* GBDT (notebook 03)", bukan "nb03" saja.
4. **Percobaan** disebut berdasarkan apa yang diubah, misalnya "model dengan bobot kelas", bukan `ce_w`. Nama kode hanya ditulis bila pembaca perlu mencarinya di tabel keluaran, dan ditulis dalam kurung.
5. **Setiap angka statistik** diikuti artinya. Contoh: "Wilcoxon p = 0,005, artinya pada sebagian besar subjek CNN lebih baik dan pola ini sulit muncul karena kebetulan."
6. **Format angka Indonesia:** koma desimal dan titik ribuan.
7. **Hindari pola tulisan otomatis:**
   - pembuka retoris ("Mari kita…", "Menariknya…");
   - pasangan tanya-jawab buatan;
   - tebal berlebihan;
   - kalimat penutup yang hanya mengulang.

## 5. Contoh

| Sebelum | Sesudah |
|---|---|
| "Konfigurasi dibekukan sesuai K7; `ce` dipakai ulang dari nb04." | "Pengaturan model sama persis dengan percobaan pada 20 subjek (notebook 04) dan tidak disetel lagi. Kalau pengaturan dipilih ulang setelah melihat hasil uji, angka akhirnya akan terlihat lebih baik daripada kenyataannya." |
| "**Input.** 153 *file* `.npz` dari notebook 02 dan `02_cache_manifest.csv`." | "**Input.** 153 rekaman yang sudah dibersihkan dan dinormalkan pada notebook 02." |
