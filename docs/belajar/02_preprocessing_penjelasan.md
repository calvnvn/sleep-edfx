# Penjelasan Notebook 02_preprocessing, Bagian A sampai D: Dari Berkas EDF ke Label per Epoch

Dokumen ini menemani `notebooks/02_preprocessing.ipynb`. Bagian E sampai J (*trimming*, penyaringan, EMG, normalisasi, pemeriksaan mutu, penyimpanan) akan ditambahkan setelah bagian-bagian itu dikerjakan.

## Bagian 1. Gambaran besar

### 1.1 Apa yang dikerjakan Fase 1

**Analogi.** Dapur restoran. Bahan mentah (berkas EDF, 7,1 GB) tidak boleh diubah. Tugas Fase 1 adalah mencuci, memotong, dan menakar bahan itu menjadi porsi siap masak, lalu menyimpannya di tempat terpisah (`data/processed/`) agar model dapat memakainya berulang kali.

**Teknis.** Satu rekaman diubah menjadi larik epoch 30 detik beserta label, dengan urutan berikut.

| Tahap | Bagian notebook | Status dokumen ini |
|---|---|---|
| Memasangkan berkas dan membaca struktur | A, B, C | Dibahas |
| Membentuk label per epoch | D | Dibahas |
| Memangkas jendela tidur (*trimming*) | E | Belum |
| Menyaring EEG dan EOG | F | Belum |
| Menangani EMG | G | Belum |
| Memotong epoch dan menormalisasi | H | Belum |
| Memeriksa mutu seluruh rekaman | I | Belum |
| Menyimpan dan memverifikasi | J | Belum |

### 1.2 Perbedaan dengan notebook EDA

Notebook 01 hanya **membaca ringkasan** (`eda_records.json`). Notebook 02 **membaca sinyal mentah** dan menghasilkan data yang diubah. Karena itu, di Bagian D kita memakai `eda_records.json` sebagai **kunci jawaban**: label yang kita bentuk sekarang harus menghasilkan hitungan yang sama dengan hasil EDA untuk seluruh 153 rekaman.

### 1.3 Aturan yang tidak boleh dilanggar

Berkas di `data/raw/` hanya dibaca, tidak pernah ditulis. Seluruh hasil disimpan di folder lain.

---

## Bagian 2. Penjelasan sel per sel

### Sel A. Persiapan dan konfigurasi

**Analogi.** Papan pengumuman di dapur: semua takaran tetap ditempel di satu tempat, sehingga koki tidak mengarang angka di tengah memasak.

**Isi.**

- `ROOT` dicari otomatis dengan menelusuri folder induk sampai menemukan `scripts/edfx_eda.py`, sehingga notebook berjalan dari folder mana pun.
- `from edfx_eda import read_header, read_signals, read_annotations` memakai ulang pembaca EDF yang sudah teruji pada 197 rekaman di tahap EDA. Kita tidak menulis ulang pembaca itu.
- Konstanta tetap: `EPOCH_S = 30`, `FS = 100`, `TRIM_EPOCHS = 60`, dan `IGNORE = -1`. Semua sel berikutnya membaca angka dari sini.
- Warna tiap tahap tidur sama dengan notebook 01, agar gambar kedua notebook terbaca sebagai satu kesatuan.

**Mengapa `IGNORE = -1`?** Ini wujud keputusan K1 (lihat `docs/catatan/KEPUTUSAN_DESAIN.md`): epoch *Unscored* dan *Movement time* tetap ada di urutan, tetapi diberi nilai −1 agar nanti tidak dihitung dalam *loss* maupun metrik.

### Sel B. Inventaris: memasangkan PSG dengan hypnogram

**Analogi.** Mencocokkan tiap kotak rekaman dengan catatan dokternya, memakai nomor di kotak.

**Cara kerja.**

1. `RAW.glob("*-PSG.edf")` mengumpulkan semua berkas sinyal.
2. Untuk tiap berkas, `psg.name[:7]` (misalnya `SC4001E`) dipakai mencari berkas hypnogram yang diawali 7 karakter yang sama. Karakter ke-8 tidak ikut karena berisi kode petugas yang bervariasi (`SC4001E0` dengan `SC4001EC`).
3. `assert len(hyp) == 1` memastikan tiap PSG memiliki **tepat satu** pasangan. Kode lama memakai `hyps[0]` tanpa memeriksa ini.
4. `subject` diambil dari karakter ke-4 dan 5, `night` dari karakter ke-6. Kode Python memakai indeks dari nol, sehingga `psg.name[3:5]` dan `psg.name[5]`.

**Pemeriksaan.** Ada 153 pasangan dan 78 subjek. Hanya subjek 13, 36, dan 52 yang memiliki satu malam; sisanya dua malam.

**Cara membaca keluaran.** Lima baris pertama menunjukkan `SC4001E` dan `SC4002E` milik subjek 0 (malam 1 dan 2). Ini alasan pembagian lipatan nanti harus berdasarkan **subjek**, bukan berkas: dua malam milik orang yang sama.

### Sel C1. Anatomi satu rekaman: header dan tata letak kanal

**Analogi.** Membaca label di kotak sebelum membukanya: isinya apa, direkam seberapa cepat, sudah disaring apa.

**Keluaran** untuk `SC4001E`: 2.650 *data record* × 30 detik = 22,08 jam, dan tabel tujuh kanal.

| Kanal | Satuan | Frekuensi | Prefilter (dari alat) |
|---|---|---|---|
| EEG Fpz-Cz | µV | 100 Hz | HP 0,5 Hz, LP 100 Hz |
| EEG Pz-Oz | µV | 100 Hz | sama |
| EOG horizontal | µV | 100 Hz | sama |
| EMG submental | µV | **1 Hz** | HP 16 Hz, rektifikasi, LP 0,7 Hz |

Kanal lain (pernapasan, suhu, penanda) tidak dipakai. Perhatikan baris EMG: nilainya sudah berupa envelope, bukan sinyal mentah. Itu sebabnya EMG diperlakukan berbeda dari EEG dan EOG di bagian berikutnya.

### Sel C2. Memuat empat sinyal

`read_signals` membaca data digital 16-bit lalu mengubahnya menjadi satuan fisik (µV) dengan skala linear dari header.

**Keluaran yang perlu diperhatikan** (`SC4001E`):

| Kanal | Sampel | Detik | Rentang |
|---|---|---|---|
| EEG Fpz-Cz | 7.950.000 | 79.500 | −192 sampai 171 µV |
| EOG horizontal | 7.950.000 | 79.500 | −512 sampai 517 µV |
| EMG submental | 79.500 | 79.500 | −0,1 sampai 3,8 |

Dua hal yang dapat dibaca dari tabel ini. Pertama, EEG 100 Hz memiliki 100 kali lebih banyak sampel daripada EMG 1 Hz, padahal durasinya sama. Kedua, amplitudo EOG sekitar tiga kali EEG. Ini salah satu alasan sinyal harus dinormalisasi per kanal.

### Sel D1. Dari rentang hypnogram ke label per epoch

**Analogi.** Dokter mencatat "tidur ringan dari menit 510 sampai 517", bukan satu catatan per 30 detik. Kita menguraikan rentang itu menjadi baris-baris absensi per 30 detik.

**Cara kerja `hypnogram_to_epochs`.**

1. `n_epochs = floor(durasi_sinyal / 30)`. Larik label **dipotong mengikuti panjang sinyal**, karena hypnogram dapat lebih panjang (150 dari 153 rekaman SC) dan label tanpa sinyal tidak berguna.
2. Larik diisi dahulu dengan `IGNORE`. Epoch yang tidak tercakup hypnogram otomatis diabaikan.
3. Untuk tiap anotasi: epoch awal `onset // 30`, epoch akhir `(onset + durasi) // 30`, lalu irisan itu diisi label hasil pemetaan.
4. Pemetaan `STAGE_TO_LABEL`: *Sleep stage 3* dan *4* digabung menjadi N3 (R&K ke AASM); *Movement time* dan *Sleep stage ?* menjadi `IGNORE`.

**Keluaran** `SC4001E`: 2.650 epoch (= ⌊79.500 ÷ 30⌋), dengan 1.997 Wake, 58 N1, 250 N2, 220 N3, 125 REM, dan 0 diabaikan. Hypnogram berakhir pada jam ke-24,0 sedangkan sinyal pada 22,1 jam; selisih itu terpotong.

**Satu trik yang sering membingungkan.** Pada hitungan, kita menulis `np.bincount(labels + 1)`. `np.bincount` hanya menerima bilangan tak negatif, jadi semua label digeser satu: −1 menjadi indeks 0 ("diabaikan"), 0 menjadi indeks 1 (Wake), dan seterusnya.

### Sel D2. Hypnogram satu malam

**Analogi.** Grafik detak jantung, tetapi untuk tahap tidur.

**Cara membaca gambar.**

- Sumbu y berurutan seperti hypnogram klinis: Wake di atas, N3 di bawah, REM di antara Wake dan N1.
- Garis berwarna adalah tahap tidur; garis abu-abu tipis menghubungkan perpindahan antar tahap.
- Area abu-abu di kanan menandai label yang melewati ujung sinyal dan dipotong.

**Temuan yang terlihat.** `SC4001E` terjaga selama 8,5 jam sebelum epoch tidur pertama dan tertidur sampai jam ke-14,5. N3 terkumpul di paruh pertama masa tidur (sekitar jam ke-9 sampai 13,7), sedangkan REM muncul berkala di sela siklus. Pola ini sesuai dengan tidur sehat, dan bagian terjaga sebelum jam ke-8,5 itulah yang akan dipangkas pada *trimming*.

### Sel D3. Sinyal pada tiga tahap

**Tujuan.** Memperlihatkan apa yang sesungguhnya akan dilihat model dalam satu epoch.

`middle_epoch_of_longest_run` mencari rentetan terpanjang suatu tahap, lalu mengambil epoch di tengahnya. Dengan begitu contoh yang dipilih adalah tahap yang stabil, bukan epoch di perbatasan.

**Cara membaca (kolom: Wake, N3, REM; baris: empat kanal).**

| Tanda | Terlihat pada | Makna |
|---|---|---|
| Gelombang lambat beramplitudo tinggi | EEG, kolom N3 | Aktivitas delta, ciri tidur dalam |
| Gerakan besar dan cepat | EOG, kolom Wake | Gerakan mata dan kedipan saat terjaga |
| Gelombang lambat beramplitudo besar | EOG, kolom REM | Gerakan mata |
| EMG sangat rendah | Baris EMG, kolom REM (sekitar 0,2 dibanding 3,5 saat Wake) | Kelumpuhan otot (*atonia*) khas REM |

Baris terakhir menjelaskan mengapa EMG dipertahankan sebagai cabang tersendiri meski hanya 30 sampel per epoch: ia pembeda REM yang paling tegas dari EEG yang mirip Wake. Pengaturan `sharey="row"` membuat tiap baris memakai skala yang sama antar kolom, sehingga perbedaan amplitudo antar tahap terlihat jujur.

### Sel D4. Label seluruh 153 rekaman dan uji silang dengan EDA

**Cara kerja.** Fungsi yang sama dijalankan pada semua rekaman, lalu dua `assert` memeriksa tiap rekaman terhadap `eda_records.json`:

1. Hitungan lima kelas (Wake sampai REM) sama dengan `full[:5]` dari EDA.
2. Jumlah epoch `IGNORE` sama dengan `full[5] + full[6]` (*Movement time* ditambah *Unscored*).

**Keluaran yang benar:**

| Kelas | Jumlah epoch |
|---|---|
| Diabaikan | 1.463 |
| Wake | 285.433 |
| N1 | 21.522 |
| N2 | 69.132 |
| N3 | 13.039 |
| REM | 25.835 |
| **Berlabel (lima kelas)** | **414.961** |
| **Seluruh epoch** | **416.424** |

Angka 414.961 adalah "SC-78 sebelum *trimming*" pada dokumen Ref-1, dan 1.463 sama dengan 1.335 (*Unscored*) + 128 (*Movement time*). Karena `assert` tidak gagal, pembentukan label kita terbukti setara dengan hasil EDA, sekaligus menegaskan bahwa belum ada epoch yang dipangkas pada tahap ini.

---

## Bagian 3. Istilah yang muncul

| Istilah | Arti dalam notebook ini |
|---|---|
| Epoch | Potongan 30 detik; satuan label dan satuan masukan model |
| Hypnogram | Berkas anotasi tahap tidur dari dokter, berupa rentang waktu |
| `pathlib.Path` | Cara modern memegang alamat berkas; operator `/` menyambung folder |
| `glob` | Mencari berkas dengan pola nama |
| `assert` | Pernyataan yang harus benar; bila salah, notebook berhenti dengan pesan galat |
| `np.diff` | Selisih elemen bersebelahan; dipakai mencari titik tempat label berubah |
| `np.bincount` | Menghitung kemunculan tiap bilangan bulat 0, 1, 2, ... |
| `sharey="row"` | Subplot satu baris berbagi skala sumbu y |

## Bagian 4. Percobaan kecil

1. Ganti `REC = "SC4001E"` dengan `"SC4762E"`, lalu jalankan ulang D1 sampai D3. Rekaman ini memiliki rentetan hampir 50 menit tanpa label; lihat bagaimana area abu-abu muncul di hypnogram.
2. Ubah `picks` di D3 menjadi `{"N1": 1, "N2": 2}` untuk melihat mengapa N1 sulit dibedakan dari Wake secara visual.
3. Hapus `+ 1` pada `np.bincount(labels + 1, ...)` dan baca pesan galatnya.

## Bagian 5. Pertanyaan untuk menguji pemahaman

1. Mengapa pasangan berkas dicocokkan dengan 7 karakter pertama, bukan seluruh nama?
2. Mengapa larik label dipotong mengikuti panjang sinyal, bukan sebaliknya?
3. Apa akibatnya bila epoch tak berlabel dibuang dari larik, bukan diberi −1?
4. Mengapa angka 414.961 di D4 belum mencerminkan *trimming*?
5. Pada D3, mengapa EMG saat REM jauh lebih rendah, dan mengapa itu berguna bagi model?
