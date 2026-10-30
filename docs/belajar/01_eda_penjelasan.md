# Penjelasan Notebook 01_eda: Memahami Kode, Alur, dan Output

Dokumen ini membantu Anda memahami `notebooks/01_eda.ipynb` sampai ke logikanya. Setiap sel dijelaskan dengan pola yang sama: **analogi → apa yang dilakukan kode → output yang Anda lihat → cara membacanya**. Semua angka di dokumen ini diambil dari output notebook Anda sendiri, yang sudah saya periksa dan sama dengan hasil uji saya.

---

## Bagian 1. Gambaran besar

### 1.1 Apakah ini pre-processing?

**Bukan.** Notebook ini adalah **EDA** (*Exploratory Data Analysis*), yaitu tahap mengamati dan mengukur data. Pre-processing adalah tahap mengubah data agar siap dipakai model. Keduanya sering tertukar, jadi ini perbedaannya.

Analogi: Anda mendapat satu truk bahan makanan untuk restoran.

| Tahap | Analogi dapur | Dalam proyek ini |
|---|---|---|
| **EDA** (sekarang) | Menimbang, menghitung, memotret, dan mencatat isi truk: berapa kilo sayur, berapa daging, mana yang busuk. **Bahan tidak diubah** | Menghitung rekaman, jam, dan epoch per tahap tidur, lalu membuat grafik. **Data tidak diubah** |
| **Pre-processing** (Fase 1, belum) | Mencuci, memotong, dan membungkus bahan menjadi porsi siap masak | Memotong sinyal jadi epoch 30 detik, memfilter, menormalisasi, menyimpan data siap latih |
| **Training** (Fase 3 dst.) | Memasak | Melatih model |

Pembedanya adalah: **apakah data berubah dan dihasilkan untuk model?**

| | EDA (notebook ini) | Pre-processing (Fase 1) |
|---|---|---|
| Menyentuh sinyal otak (EEG/EOG/EMG)? | Tidak | Ya (filter, normalisasi) |
| Menghasilkan data untuk model? | Tidak | Ya |
| Keluaran | Tabel dan grafik untuk manusia | Berkas data untuk mesin |
| Fungsi | Memahami kondisi data | Menyiapkan data |

Satu hal yang membuatnya membingungkan: di notebook Anda ada kata **"trimming"**. Di sini *trimming* hanya **dihitung**: kita menghitung berapa epoch yang *akan* tersisa bila data dipangkas ±30 menit. **Tidak ada data yang benar-benar dipangkas atau disimpan.** *Trimming* yang sebenarnya dilakukan di Fase 1.

### 1.2 Alur data dari awal sampai gambar

```
[A] Berkas EDF mentah                      data/raw/sleep-edfx/
    197 PSG (sinyal) + 197 hypnogram (label dokter)        (8 GB, tidak diubah)
            │
            │   dibaca oleh  scripts/edfx_eda.py     (dijalankan sekali)
            ▼
[B] Kartu ringkasan per rekaman            results/eda/eda_records.json
    197 baris, hanya angka ringkasan                       (127 KB)
            │
            │   dibaca oleh  notebooks/01_eda.ipynb  (cepat, boleh diulang)
            ▼
[C] Tabel dan grafik                       results/tables/ dan results/figures/
```

Mengapa dibuat dua tahap? Membaca 8 GB berkas EDF itu lambat dan hanya perlu sekali. Hasilnya diringkas ke berkas kecil (B), sehingga notebook (C) bisa dijalankan berulang kali dalam hitungan detik tanpa menyentuh data besar.

---

## Bagian 2. Apa itu `eda_records.json`?

### 2.1 Analogi

Bayangkan sebuah klinik tidur menyimpan **197 map rekam medis**, satu map per malam rekaman. Tiap map berisi ribuan halaman grafik gelombang otak. Seorang asisten lalu membuka satu per satu map itu dan menulis **satu kartu ringkasan per map**: nama, usia, berapa jam rekamannya, dan berapa lama pasien berada di tiap tahap tidur. `eda_records.json` adalah **tumpukan 197 kartu ringkasan** itu. Kartu bukan isi map; ia hanya catatan hasil hitungan.

### 2.2 Isi satu kartu (contoh nyata: rekaman pertama, `SC4001E`)

| Isi kartu | Nilai | Arti |
|---|---|---|
| `rec` | SC4001E | Kode rekaman (subjek 00, malam 1) |
| `set` | cassette | Sub-dataset SC |
| `subject` | SC00 | ID subjek |
| `night` | 1 | Malam ke-1 |
| `sex`, `age` | Female, 33 | Jenis kelamin dan usia |
| `hours` | 22,08 | Durasi sinyal rekaman (jam) |
| `labels`, `fs` | 7 nama kanal dan 7 laju sampel | Daftar kanal (EEG, EOG, EMG, ...) beserta lajunya |
| `full` | [1997, 58, 250, 220, 125, 0, 0] | Jumlah epoch per kelas **sebelum** *trimming* |
| `trim` | [188, 58, 250, 220, 125, 0, 0] | Jumlah epoch per kelas **sesudah** *trimming* (hasil hitungan) |

Tujuh angka pada `full` dan `trim` berurutan: **Wake, N1, N2, N3, REM, Movement time, Unscored**.

### 2.3 Cara membaca contoh itu

- Wake sebelum *trimming* adalah 1.997 epoch dan sesudahnya 188. Selisihnya 1.809 epoch × 30 detik ≈ **15,1 jam Wake yang akan dibuang**. Itulah sebagian besar siang hari yang terekam oleh alat kaset.
- Tahap tidur (N1 + N2 + N3 + REM) = 58 + 250 + 220 + 125 = 653 epoch, tidak berubah sebelum maupun sesudah.
- Sisa 188 epoch Wake itu terdiri dari margin 30 menit di awal dan akhir tidur ditambah Wake yang terselip di tengah malam.
- Total setelah *trimming* = 841 epoch = **7,0 jam**, sehingga rekaman 22 jam menyusut menjadi satu malam yang wajar.

### 2.4 Apa yang **tidak** ada di dalamnya

Berkas ini **tidak memuat sinyal otak**, dan juga tidak memuat label epoch demi epoch. Karena itu ia cukup untuk membuat tabel dan grafik ringkasan, tetapi **tidak cukup** untuk melatih model. Untuk model, kita perlu membaca ulang EDF di Fase 1.

### 2.5 Dari mana tiap isian kartu diambil (contoh nyata `SC4001E`)

Satu kartu dibuat dari **tiga sumber**: nama berkas, *header* berkas PSG, dan berkas hypnogram. Isi sinyal (±48 MB per rekaman) tidak disentuh.

| Isian | Nilai contoh | Diambil dari | Cara mendapatkannya |
|---|---|---|---|
| `rec` | SC4001E | Nama berkas `SC4001E0-PSG.edf` | 7 karakter pertama |
| `set` | cassette | Nama folder | `sleep-cassette` → cassette |
| `subject` | SC00 | Nama berkas | Karakter ke-4 dan 5 (`00`) |
| `night` | 1 | Nama berkas | Karakter ke-6 (`1`) |
| `sex`, `age` | Female, 33 | Header PSG, bidang *patient* (byte 8–88) | Isinya `X F X Female_33yr`; dicari dengan pola `Female_33yr` |
| `hours` | 22,08 | Header PSG, byte 236–252 | 2.650 *record* × 30 detik = 79.500 detik ÷ 3.600 |
| `start` | 16.13.00 | Header PSG, byte 176–184 | Jam rekaman dimulai |
| `labels` | 7 nama kanal | Header kanal PSG | Label 16 byte per kanal |
| `fs` | 100, 100, 100, 1, 1, 1, 1 | Header kanal PSG | Sampel per *record* ÷ 30 detik |
| `prefilter`, `dims` | "HP:16Hz Rectification LP:0.7Hz", "uV" | Header kanal PSG | Teks apa adanya |
| `ann_end_h` | 24,0 | Hypnogram | Anotasi terakhir berakhir pada jam ke-24 |
| `full` | [1997, 58, 250, 220, 125, 0, 0] | Hypnogram | Anotasi dipecah menjadi epoch 30 detik lalu dihitung |
| `trim` | [188, 58, 250, 220, 125, 0, 0] | Hypnogram | Sama, tetapi hanya epoch dalam jendela ±30 menit dari tidur |

Hypnogram bukan daftar label per epoch, melainkan **daftar rentang**. Contoh dari berkas ini (detik mulai, durasi, label): `(0, 30630, Wake)`, `(30630, 120, N1)`, `(30750, 390, N2)`, `(31140, 30, N3)`, ... Entri terakhir adalah `(79500, 6900, ?)`, yaitu pelabelan setelah sinyal berakhir. Entri itu **terpotong** karena jumlah epoch mengikuti panjang sinyal (⌊79.500 ÷ 30⌋ = 2.650), dan itulah sebabnya `ann_end_h` (24,0 jam) lebih besar dari `hours` (22,08 jam).

**Satu malam rekaman SC4001E, dari awal sampai akhir (2.650 epoch):**

```
epoch:   0 ───────────── 1021 ═════ tidur ═════ 1741 ──────────── 2649
          [   Wake siang/sore   ][  N1 N2 N3 REM  ][  Wake pagi/siang  ]
                           ▲ tidur pertama      ▲ tidur terakhir
trim:                 [961 ◄ 60 epoch ┤ 1021 .. 1741 ├ 60 epoch ► 1801]
                      └──────── 841 epoch (7,0 jam) dipertahankan ────────┘
```

- Sebelum epoch 1021 ada **1.021 epoch Wake** (±8,5 jam), dan sesudah epoch 1741 ada **908 epoch Wake**. Itu terjadi karena alat tetap merekam di luar jam tidur.
- Sisanya, 68 epoch Wake, terselip di tengah malam saat pasien terbangun. Totalnya 1.021 + 908 + 68 = **1.997 epoch Wake** (angka `full_Wake`).
- *Trimming* mempertahankan epoch 961 sampai 1801: 60 epoch Wake sebelum tidur, seluruh isi malam, dan 60 epoch Wake sesudahnya. Jadi Wake yang tersisa = 60 + 60 + 68 = **188 epoch** (angka `trim_Wake`).
- Bila `start` = 16.13 dan epoch tidur pertama dimulai 30.630 detik (8 jam 30 menit) kemudian, pasien tertidur sekitar pukul 00.43 dan terbangun sekitar pukul 06.44, sementara rekaman terus berjalan sampai sekitar pukul 14.18. (Jam ini hitungan saya dari `start` dan hypnogram.)

### 2.6 Bagaimana *trimming* dihitung tanpa pre-processing?

**Pertanyaan.** *Trimming* adalah langkah pre-processing, tetapi `eda_records.json` sudah memuat angka `trim`. Bagaimana angka itu diperoleh sebelum Fase 1?

**Jawaban singkat.** Yang dipangkas di EDA hanyalah **larik label** (satu angka per epoch), bukan sinyal. Larik itu dapat dibentuk dari hypnogram saja, sehingga tidak ada satu pun sampel sinyal yang dibaca, diubah, atau disimpan.

**Analogi.** Buku absensi satu malam berisi 2.650 baris, satu baris per 30 detik. Untuk mengetahui berapa baris yang tersisa jika bagian "terjaga" di awal dan akhir dibuang, cukup menunjuk baris pertama dan terakhir yang bertuliskan "tidur", lalu menghitung. Rekaman otaknya tidak perlu dibuka. EDA adalah tahap *mengukur*; Fase 1 adalah tahap *memotong*.

**Alur lima langkah** (fungsi di `scripts/edfx_eda.py`, contoh `SC4001E`):

| Langkah | Fungsi | Yang dibaca | Hasil pada `SC4001E` |
|---|---|---|---|
| 1. Memasangkan berkas | `glob`, `nama[:7]` | Nama berkas saja | `SC4001E0-PSG.edf` dengan `SC4001EC-Hypnogram.edf` |
| 2. Membaca *header* PSG | `read_header` | 256 byte utama dan blok kanal (bukan data sinyal) | 7 kanal, 79.500 detik |
| 3. Membaca hypnogram | `read_annotations` | Berkas 4,6 KB | 154 anotasi `(onset, durasi, teks)` |
| 4. Membentuk label per epoch | `epochs_from_ann` | Anotasi dan durasi PSG | Larik 2.650 angka |
| 5. Memangkas dan menghitung | `trim_wake`, `np.bincount` | Larik label | 841 epoch tersisa |

**Langkah 4: dari rentang menjadi label per epoch.** Hypnogram menyimpan rentang waktu, bukan label per epoch. Untuk tiap anotasi, epoch awal adalah `onset // 30` dan epoch akhir adalah `(onset + durasi) // 30`. Rentang pertama (0 sampai 30.630 detik, Wake) mengisi epoch 0 sampai 1020, yaitu 1.021 epoch. Larik dibuat sepanjang `⌊durasi sinyal ÷ 30⌋`, sehingga anotasi yang melewati ujung sinyal dipotong oleh `min(b, n)`.

**Langkah 5: pemangkasan.** Dua baris kode menjalankannya:

```python
sleep = np.where((lab >= 1) & (lab <= 4))[0]          # indeks epoch yang bukan Wake
lab[max(0, sleep[0] - 60) : min(len(lab), sleep[-1] + 61)]
```

Angka 60 berasal dari 30 menit ÷ 0,5 menit per epoch. Pada `SC4001E`, epoch tidur pertama adalah 1021 dan yang terakhir 1741, sehingga jendela yang dipertahankan adalah epoch 961 sampai 1801 (841 epoch). Wake di dalam jendela: 60 + 68 + 60 = 188; Wake yang terbuang: 1.997 − 188 = 1.809 epoch.

**Apa yang disebut "filtrasi" di skrip ini.** Ada tiga hal, dan tidak ada yang membuang rekaman:

1. **Pemetaan label.** Teks anotasi yang tidak ada di tabel `STAGE` diabaikan. "Sleep stage 3" dan "Sleep stage 4" digabung menjadi N3. "Movement time" dan "Sleep stage ?" ditampung di kolom tersendiri.
2. **Jendela *trimming*.** Satu irisan larik, `lab[a:b]`.
3. **Pemilahan SC dan ST.** Dilakukan di notebook (Sel 2), bukan di skrip. Seluruh 197 rekaman masuk ke JSON.

Larik label hasil pangkasan (`lab_trim`) **tidak disimpan**: baris terakhir `main()` membuangnya dengan `if k != "lab_trim"`. Larik itu hanya dipakai sebentar di memori untuk matriks transisi. Inilah salah satu alasan JSON hanya 127 KB.

**Pembeda EDA dan Fase 1.**

| | EDA (sekarang) | Fase 1 (nanti) |
|---|---|---|
| Aturan ±30 menit dipakai | Ya, pada **label** | Ya, pada **label dan sinyal** |
| Sinyal otak dibaca | Tidak | Ya |
| Hasil | Angka hitungan | Berkas data siap latih |
| Fungsi angka `trim` | Mengukur dampak aturan | Kunci jawaban: total SC-78 harus **195.469** epoch |

Aturan ±30 menit sendiri berasal dari rancangan Ref-1, bukan dari hasil EDA. EDA hanya menghitung akibat penerapannya.

**Catatan keterbatasan.** Skrip memasangkan berkas dengan `hyps[0]` tanpa memeriksa jumlah hasil `glob`. Pada dataset ini aman, karena tiap PSG memiliki tepat satu hypnogram dan hasil 195.469 epoch cocok dengan verifikasi mandiri. Pada dataset lain, pengecekan eksplisit diperlukan.

---

## Bagian 3. Penjelasan sel per sel

Notebook Anda memiliki **8 sel kode**. Di bawah ini disebut Sel 1 sampai Sel 8 sesuai urutannya.

Cara membaca sel yang tepat: tanyakan tiga hal. **Apa masukannya? Apa yang diubah? Apa keluarannya?**

### Peta isi memori setelah tiap sel

Variabel yang "hidup" setelah sel dijalankan. Bila variabel hilang atau galat `NameError`, kemungkinan ada sel sebelumnya yang belum dijalankan.

| Setelah sel | Variabel baru yang tersedia |
|---|---|
| Sel 1 | `STAGES`, `EPOCH_MIN`, warna, gaya grafik |
| Sel 2 | `ROOT`, `df`, `FULL_COLS`, `TRIM_COLS`, `sc`, `sc20`, `st` |
| Sel 3 | (tidak ada, hanya memeriksa) |
| Sel 4 | `overview`, `summary` |
| Sel 5 | `class_share`, `before`, `after` |
| Sel 6 | `minutes`, `stats` |
| Sel 7 | (hanya menggambar) |
| Sel 8 | `share`, `n_rec` |

---

### Sel 1: Setup

**Analogi:** menyiapkan meja kerja sebelum mulai: mengambil alat (pustaka), menaruh daftar tetap yang akan dipakai berulang (urutan tahap tidur dan warna), dan menyetel gaya kerja (gaya grafik) agar semua hasil seragam.

**Apa yang dilakukan kode:**

| Bagian | Fungsi |
|---|---|
| `import ...` | Memanggil alat: `pandas` untuk tabel, `numpy` untuk hitungan angka, `matplotlib` untuk grafik |
| `STAGES = [...]` | Urutan lima tahap tidur. Dipakai di seluruh notebook agar selalu sejajar |
| `EPOCH_MIN = 0.5` | Satu epoch = 30 detik = 0,5 menit. Dipakai untuk mengubah epoch menjadi menit |
| `STAGE_COLORS`, `ACCENT`, `MUTED` | Daftar warna tetap |
| `plt.rcParams.update(...)` | Mengatur gaya semua grafik sekali (latar, garis kisi, font, ukuran judul) |

**Output:** tidak ada. Sel ini hanya menyiapkan. Tidak adanya output itu normal.

---

### Sel 2: Memuat dan menyiapkan data

**Analogi:** mengambil tumpukan 197 kartu ringkasan dari laci, menyusunnya menjadi satu tabel rapi, lalu memisahkannya ke dalam tiga tumpukan: kartu SC, kartu SC-20, dan kartu ST.

**Apa yang dilakukan kode, baris demi baris:**

1. **`ROOT = next(...)`**: mencari folder utama proyek. Kode naik dari folder tempat notebook berada sampai menemukan folder yang berisi `results/eda`. Dengan cara ini, notebook tetap jalan walau dijalankan dari folder lain.
2. **`FIG_DIR`, `TAB_DIR`** dan `mkdir(...)`: menetapkan folder simpanan grafik dan tabel, lalu membuatnya bila belum ada.
3. **`pd.DataFrame(json.load(f))`**: membaca berkas JSON lalu mengubahnya menjadi **tabel** (*DataFrame*) 197 baris. Ibaratnya lembar Excel yang tiap barisnya satu rekaman.
4. **Loop `for tag in ("full", "trim")`**: di kartu, hitungan epoch berbentuk satu daftar `[1997, 58, 250, 220, 125, 0, 0]`. Daftar seperti itu sulit dipakai, jadi kode **memecahnya menjadi tujuh kolom bernama**:
   - `df.pop(tag)` mengambil kolom `full` keluar dari tabel,
   - `.tolist()` mengubahnya menjadi daftar-daftar,
   - `pd.DataFrame(..., columns=LABELS)` membuat tabel baru dengan tujuh kolom,
   - `.add_prefix("full_")` menambah awalan sehingga namanya menjadi `full_Wake`, `full_N1`, dan seterusnya,
   - `pd.concat([df, counts], axis=1)` menempelkan kolom baru itu ke samping tabel utama.
5. **`df["group"] = np.where(...)`**: membuat kolom baru: bila `set` bernilai `cassette` maka `SC`, selain itu `ST`.
6. **`df["subject_no"] = df["subject"].str[2:].astype(int)`**: mengambil angka dari ID, misalnya `"SC00"` menjadi `0`. Gunanya untuk membentuk SC-20.
7. **Tiga baris pemfilteran:**
   - `sc = df[df["group"] == "SC"]` mengambil baris yang kolom group-nya `SC` → **153** rekaman,
   - `sc20 = sc[sc["subject_no"] < 20]` mengambil subjek bernomor 0–19 → **39** rekaman,
   - `st = ...` mengambil kelompok ST → **44** rekaman.

**Output Anda:** `(197, 29)`

Artinya tabel utama punya **197 baris** (rekaman) dan **29 kolom**. Asal 29 kolom: 13 kolom asli + 7 kolom `full_*` + 7 kolom `trim_*` + `group` + `subject_no`.

**Contoh isi tabel setelah sel ini (tiga baris pertama, kolom terpilih):**

| rec | hours | full_Wake | full_N1 | full_N2 | full_N3 | full_REM | trim_Wake | trim_N1 | trim_N2 | trim_N3 | trim_REM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SC4001E | 22,08 | 1997 | 58 | 250 | 220 | 125 | 188 | 58 | 250 | 220 | 125 |
| SC4002E | 23,58 | 1885 | 59 | 373 | 297 | 215 | 183 | 59 | 373 | 297 | 215 |
| SC4011E | 23,35 | 1856 | 109 | 562 | 105 | 170 | 157 | 109 | 562 | 105 | 170 |

---

### Sel 3: Pemeriksaan kewarasan (*sanity check*)

**Analogi:** sebelum menandatangani laporan, kasir mencocokkan total uang di laci dengan total di buku. Bila selisih, berhenti dan cari penyebabnya sebelum lanjut.

**Apa yang dilakukan kode:**
- `assert len(df) == 197 and ...` memeriksa bahwa jumlah rekaman memang 197, 153, 44, dan 39.
- `assert {...} == expected` menghitung jumlah epoch tiap kelas pada SC setelah *trimming*, lalu mencocokkannya dengan angka yang sudah kita verifikasi dari sumber lain: Wake 65.941, N1 21.522, N2 69.132, N3 13.039, REM 25.835.
- `assert` bekerja begini: **bila benar, diam; bila salah, notebook berhenti dengan galat `AssertionError`.**

**Output Anda:** `All checks passed`

**Mengapa penting:** kesalahan terburuk dalam analisis data adalah yang **tidak berbunyi**. Bila ada satu kolom salah, grafik tetap tampak normal tetapi menyesatkan. `assert` membuat kesalahan seperti itu berbunyi.

---

### Sel 4: Tabel ringkasan dataset

**Analogi:** satu halaman "profil perusahaan" di awal laporan: berapa cabang, berapa karyawan, berapa omzet. Pembaca cukup membaca halaman ini untuk tahu skala data.

**Apa yang dilakukan kode:**
- `def overview(d):` mendefinisikan sebuah **resep**: diberi sekumpulan rekaman `d`, ia menghitung enam angka (jumlah rekaman, subjek unik, rentang usia, rata-rata durasi, total durasi, dan total epoch berlabel).
- Resep itu lalu dipanggil **empat kali**, untuk SC-78, SC-20, ST, dan seluruh data. Hasilnya digabung menjadi satu tabel `summary`.
- `to_csv(...)` menyimpan tabel itu ke `results/tables/01_dataset_overview.csv`.
- Baris terakhir `summary` memunculkan tabelnya di bawah sel.

**Output Anda:**

| | SC-78 | SC-20 | ST | All |
|---|---|---|---|---|
| Recordings | 153 | 39 | 44 | 197 |
| Subjects | 78 | 20 | 22 | 100 |
| Age range (yr) | 25-101 | 25-34 | 18-78 | 18-101 |
| Mean duration (h) | 22,7 | 22,9 | 8,6 | 19,5 |
| Total duration (h) | 3470,2 | 894,8 | 378,8 | 3849,0 |
| Labelled epochs, trimmed | 195469 | 42307 | 42467 | 237936 |

**Cara membaca:** SC rata-rata merekam **22,7 jam** per rekaman, sedangkan ST hanya **8,6 jam**. Selisih sebesar itu adalah petunjuk pertama bahwa keduanya adalah dua dunia yang berbeda (SC direkam di rumah, ST di laboratorium).

**Mengapa memakai resep (fungsi), bukan menulis ulang empat kali:** bila ada koreksi, cukup mengubah satu tempat.

---

### Sel 5: Distribusi kelas sebelum dan sesudah *trimming* (Grafik 1)

**Analogi:** bayangkan Anda ingin tahu apakah sebuah kelas siswa seimbang. Sebelum memangkas, **dua pertiga isi kelas adalah murid yang cuma duduk diam** (Wake). Setelah mengeluarkan murid yang tidak relevan, komposisi kelas menjadi jauh lebih seimbang.

**Apa yang dilakukan kode:**
1. `class_share(d, cols)` menghitung persentase tiap tahap:
   - `d[cols].sum()` menjumlahkan epoch tiap tahap **dari semua rekaman** (jumlah per kolom),
   - `c / c.sum() * 100` membagi dengan total lalu menjadikannya persen,
   - `.set_axis(STAGES)` mengganti nama baris dari `full_Wake` menjadi `Wake`, dan seterusnya.
2. Fungsi itu dipanggil dua kali: pakai kolom `full_*` (sebelum) dan kolom `trim_*` (sesudah).
3. Sisa kode menggambar dua batang per tahap. Abu-abu untuk "sebelum" dan biru untuk "sesudah", agar mata langsung tertuju ke biru yang menjadi data kita.
4. Angka persen hanya ditulis di pasangan **Wake**, karena itulah pesan grafik ini.

**Output Anda:**

| Tahap | Sebelum (%) | Sesudah (%) |
|---|---|---|
| Wake | 68,8 | 33,7 |
| N1 | 5,2 | 11,0 |
| N2 | 16,7 | 35,4 |
| N3 | 3,1 | 6,7 |
| REM | 6,2 | 13,2 |

**Cara membaca:**
- Tanpa *trimming*, **Wake menguasai 68,8%** data. Model yang selalu menjawab "Wake" sudah benar hampir 70% tanpa belajar apa pun.
- Setelah *trimming*, N2 menjadi kelas terbesar (35,4%) dan N3 terkecil (6,7%). Rasio terbesar ke terkecil sekitar 5:1, ketimpangan sedang.
- Persentase N1, N2, N3, dan REM semuanya naik hanya karena Wake dikurangi. Jumlah epoch tidurnya sendiri tidak berubah.

---

### Sel 6: Rata-rata waktu per tahap (Grafik 2)

**Analogi:** hitung rata-rata berapa jam seorang karyawan menghabiskan waktu untuk rapat, email, dan istirahat dalam sehari. Rata-rata saja belum cukup; Anda juga ingin tahu apakah semua karyawan mirip atau ada yang jauh berbeda.

**Apa yang dilakukan kode:**
1. `sc[TRIM_COLS].mul(EPOCH_MIN)` mengambil lima kolom `trim_*` dan **mengalikan setiap angka dengan 0,5** → epoch menjadi **menit**. `.set_axis(STAGES, axis=1)` mengganti nama kolom.
2. `minutes.agg(["mean", "std", "median"])` menghitung tiga ukuran **per tahap**:
   - **mean** = rata-rata,
   - **std** = simpangan baku (seberapa jauh data biasanya menyimpang dari rata-rata),
   - **median** = nilai tengah bila semua rekaman diurutkan.
3. `.T` memutar tabel (baris menjadi kolom) supaya tiap baris mewakili satu tahap.
4. Grafik digambar dari tiga lapis:
   - **batang biru muda** = rata-rata,
   - **garis biru dengan penutup** = rata-rata ± 1 simpangan baku,
   - **titik abu-abu** = **setiap rekaman**, 153 titik per tahap.
5. `rng.uniform(-0.18, 0.18, ...)` menggeser titik sedikit ke kiri dan kanan secara acak (*jitter*) agar tidak menumpuk. Angka acak diberi *seed* 0 supaya gambar sama setiap kali dijalankan.

**Output Anda:**

| Tahap | Rata-rata (menit) | Std | Median |
|---|---|---|---|
| Wake | 215,5 | 182,1 | 140,5 |
| N1 | 70,3 | 45,9 | 59,0 |
| N2 | 225,9 | 56,5 | 224,0 |
| N3 | 42,6 | 40,6 | 39,0 |
| REM | 84,4 | 31,5 | 84,5 |

**Contoh nyata satu rekaman** (SC4001E, baris pertama `minutes`): Wake 94,0 menit, N1 29,0, N2 125,0, N3 110,0, REM 62,5. Angka itu berasal dari 188 × 0,5 = 94,0 dan seterusnya.

**Cara membaca, dan ini bagian terpenting:**
- **N2 dan REM stabil.** Rata-rata hampir sama dengan median (226 vs 224; 84,4 vs 84,5), dan simpangan bakunya kecil. Hampir semua orang memiliki jumlah yang mirip.
- **Wake tidak stabil.** Rata-rata (215,5) jauh lebih besar dari median (140,5), dan simpangan bakunya (182) hampir sama dengan rata-ratanya. Artinya **mayoritas rekaman Wake-nya sedang (sekitar 140 menit), tetapi beberapa rekaman Wake-nya sangat panjang (sampai sekitar 900 menit)**, dan rekaman ekstrem itu menarik rata-rata ke atas. Distribusi seperti ini disebut **miring ke kanan** (*right-skewed*).
- Karena itu, saat ditanya "berapa rata-rata Wake?", jawaban yang jujur adalah: **"rata-rata 216 menit, median 140 menit; angkanya sangat bervariasi antar-rekaman."**
- Rata-rata ini dihitung **per rekaman**, bukan per subjek.

---

### Sel 7: Durasi rekaman, SC vs ST (Grafik 3)

**Analogi:** dua kelompok pekerja diukur lama kerjanya. Satu kelompok bekerja shift semalam suntuk, kelompok lain hanya shift pendek. Histogram menyusun mereka dalam "kotak lebar jam" untuk melihat tumpukannya.

**Apa yang dilakukan kode:**
- `bins = np.arange(7, 25, 1)` membuat kotak berlebar 1 jam dari 7 sampai 24 jam.
- `ax.hist([sc["hours"], st["hours"]], stacked=True, ...)` menghitung berapa rekaman jatuh di tiap kotak untuk SC dan ST, lalu **menumpuk** keduanya dalam batang yang sama dengan warna berbeda agar tidak saling menutupi.
- Judul dibuat otomatis dari data (`sc['hours'].median()`), sehingga angka di judul selalu sesuai isi grafik.

**Output Anda:** satu grafik, tanpa tabel. Judulnya menyatakan median SC **23,0 jam** vs ST **8,6 jam**.

**Cara membaca:** ada **dua kelompok yang terpisah jauh**: ST menumpuk di sekitar 8–10 jam (semalam di laboratorium), SC menumpuk di sekitar 21–24 jam (direkam seharian di rumah). Itulah penyebab visual mengapa SC punya porsi Wake yang besar, dan mengapa SC perlu dipangkas sedangkan ST hampir tidak.

---

### Sel 8: Komposisi tahap menurut usia (Grafik 4)

**Analogi:** membandingkan isi piring makan siang di empat kelompok usia. Total piring selalu 100%, yang berubah adalah proporsi nasi, lauk, dan sayur.

**Apa yang dilakukan kode:**
1. `pd.cut(sc["age"], bins=[24,39,59,79,120], labels=[...])` memasukkan tiap subjek ke satu kelompok usia: 25-39, 40-59, 60-79, 80+. Batas bawah dibuat satu angka di bawah awal rentang karena `pd.cut` memakai batas kanan.
2. `groupby("age_group")` mengelompokkan baris menurut kelompok usia, lalu `.sum()` menjumlahkan epoch per tahap di dalam tiap kelompok.
3. `share.div(share.sum(axis=1), axis=0) * 100` membagi tiap baris dengan totalnya sehingga tiap kelompok berjumlah 100%.
4. Loop `for s in STAGES` menggambar batang bertumpuk: tiap tahap digambar **di atas** tahap sebelumnya. Variabel `bottoms` menyimpan ketinggian dasar yang terus bertambah.
5. Angka persen ditulis di dalam segmen, **hanya bila segmen ≥ 6%** (segmen yang lebih kecil tidak muat).

**Output Anda:**

| Kelompok usia | n rekaman | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|
| 25-39 | 39 | 19,6% | 6,6% | 42,1% | 13,5% | 18,2% |
| 40-59 | 41 | 33,2% | 10,3% | 37,7% | 5,5% | 13,4% |
| 60-79 | 41 | 31,2% | 13,7% | 35,9% | 6,2% | 13,0% |
| 80+ | 32 | 48,0% | 12,6% | 27,3% | 2,8% | 9,3% |

**Cara membaca:** seiring usia, **N3 (tidur dalam) menyusut dari 13,5% menjadi 2,8%**, sedangkan Wake naik dari 19,6% menjadi 48,0%. Tidur menjadi lebih ringan dan lebih terputus-putus. Dampaknya pada model: kualitas prediksi kemungkinan berbeda antar-kelompok usia, dan itu salah satu alasan mengapa skor per subjek bervariasi.

---

## Bagian 4. Istilah pandas yang muncul di kode

Daftar singkat agar Anda bisa membaca kode sendiri.

| Istilah | Makna sederhana |
|---|---|
| **DataFrame** | Tabel seperti lembar Excel: baris = rekaman, kolom = atribut |
| **Series** | Satu kolom (atau satu baris) dari tabel |
| `df["kolom"]` | Mengambil satu kolom |
| `df[df["group"] == "SC"]` | **Menyaring** baris: hanya yang kolom group-nya SC |
| `.sum()`, `.mean()`, `.median()`, `.std()` | Jumlah, rata-rata, nilai tengah, simpangan baku |
| `.pop("nama")` | Mengambil sebuah kolom keluar dari tabel (kolom itu hilang dari tabel asal) |
| `.tolist()` | Mengubah isi kolom menjadi daftar Python biasa |
| `.add_prefix("x_")` | Menambah awalan pada nama semua kolom |
| `pd.concat([a, b], axis=1)` | Menempel dua tabel **ke samping** |
| `.set_axis(nama)` | Mengganti nama baris atau kolom |
| `.groupby("kolom")` | Membagi tabel menjadi kelompok menurut isi satu kolom |
| `.agg([...])` | Menghitung beberapa ukuran sekaligus |
| `.T` | Memutar tabel (baris ↔ kolom) |
| `pd.cut(angka, bins, labels)` | Mengubah angka menjadi kelompok rentang (contoh usia → "25-39") |

## Bagian 5. Percobaan kecil untuk memperdalam pemahaman

Tambahkan sel baru di bawah Sel 8 dan jalankan satu per satu. Tujuannya melihat isi variabel secara langsung, bukan mengubah analisis. Semuanya hanya membaca data, tidak mengubahnya.

| Ketik di sel | Yang akan Anda lihat |
|---|---|
| `df.head(3)` | Tiga baris pertama tabel utama |
| `df.columns.tolist()` | Daftar 29 kolom |
| `df["group"].value_counts()` | Jumlah rekaman: SC 153, ST 44 |
| `sc.shape` | `(153, 29)`: tabel SC |
| `minutes.head(3)` | Menit per tahap untuk tiga rekaman pertama |
| `sc["age"].describe()` | Ringkasan usia SC: rata-rata 59,1, rentang 25–101 |

Cara berpikir: bila sebuah baris kode terasa membingungkan, **cetak variabel sebelum dan sesudahnya**. Perbedaan keduanya adalah jawaban dari apa yang dilakukan baris itu.

## Bagian 6. Mengapa EDA di sini belum pre-processing, dan apa selanjutnya

Yang sudah Anda kerjakan: **memahami kondisi data**. Yang belum: mengubah data menjadi bahan model.

| Pertanyaan | EDA (selesai) | Fase 1 (berikutnya) |
|---|---|---|
| Berapa rekaman dan jam? | Sudah dijawab | |
| Kelas mana yang timpang? | Sudah dijawab | |
| Apa efek *trimming*? | Sudah **dihitung** | *Trimming* benar-benar **dilakukan** |
| Seperti apa wujud sinyal tiap tahap? | Belum | Membaca sinyal, memotong per epoch |
| Bagaimana sinyal disiapkan untuk model? | Belum | Filter, normalisasi per rekaman |

Fase 1 akan memakai pemahaman dari EDA ini sebagai **kunci jawaban**: setelah *trimming* dilakukan sungguhan, jumlah epoch SC-78 harus tepat **195.469** (Wake 65.941, N1 21.522, N2 69.132, N3 13.039, REM 25.835). Itu persis angka yang diperiksa `assert` di Sel 3.

## Bagian 7. Cara menyampaikan ke dosen

1. **Profil dataset (Sel 4):** "Ada 197 rekaman dari 100 subjek. Kami memakai SC, yaitu 153 rekaman dari 78 subjek, rata-rata 22,7 jam per rekaman."
2. **Mengapa *trimming* (Grafik 1):** "Tanpa *trimming*, 68,8% data adalah Wake siang hari. Setelah ±30 menit menjadi 33,7%."
3. **Rata-rata waktu per tahap (Grafik 2):** "Per rekaman setelah *trimming*: N2 ±226 menit, Wake ±216, REM ±84, N1 ±70, N3 ±43. Wake sangat bervariasi: mediannya 140 menit."
4. **Karakter data (Grafik 3 dan 4):** "SC direkam di rumah ±23 jam, ST di laboratorium ±9 jam. N3 menyusut dari 13,5% pada usia 25–39 menjadi 2,8% pada usia 80+."

## Bagian 8. Keterbatasan

- Rata-rata dihitung **per rekaman**, bukan per subjek.
- Wake berdistribusi sangat miring, sehingga rata-rata ± simpangan baku bukan ringkasan yang ideal. Sebutkan median.
- "Kondisi" ditafsirkan sebagai tahap tidur. Tafsiran ini perlu dikonfirmasi ke dosen.
- Belum tercakup: durasi rata-rata satu episode, contoh hypnogram, contoh sinyal, dan matriks transisi. Semuanya butuh membaca berkas EDF.

## Bagian 9. Lampiran teknis: pustaka dan prinsip visualisasi

**Pustaka (versi di `.venv` Anda):** `pandas` 3.1.0 (tabel), `numpy` 2.4.6 (hitungan angka), `matplotlib` 3.11.2 (grafik), ditambah `json` dan `pathlib` bawaan Python. `seaborn` terpasang tetapi tidak dipakai karena `matplotlib` sudah cukup; `plotly` tidak dipakai karena kita memerlukan gambar statis untuk laporan dan tidak tercantum di `requirements.txt`.

**Prinsip yang diterapkan pada grafik:**
- Satu pesan per grafik, dan **judul memuat temuannya**.
- **Warna tetap per entitas:** tiap tahap selalu berwarna sama pada grafik bertumpuk (Wake biru, N1 oranye, N2 hijau, N3 kuning, REM merah muda).
- **Teknik penekanan:** warna aksen biru untuk yang menjadi fokus, abu-abu untuk konteks.
- **Label selektif:** angka hanya ditulis bila perlu. Segmen yang terlalu kecil tidak diberi teks agar tidak meluber.
- **Palet sudah divalidasi** dengan alat uji: lolos uji keterbedaan bagi penderita buta warna dan penglihatan normal. Satu peringatan (kontras warna aqua, kuning, dan magenta di bawah 3:1) dikompensasi dengan angka persen di dalam segmen dan tabel pendamping.

## Bagian 10. Pertanyaan untuk menguji pemahaman

1. Apa perbedaan EDA dan pre-processing? Mengapa notebook ini termasuk EDA padahal ada kata "trimming"?
2. Apa yang ada dan **tidak ada** di dalam `eda_records.json`? Mengapa berkas itu tidak cukup untuk melatih model?
3. Di Sel 2, mengapa daftar `[1997, 58, 250, ...]` dipecah menjadi tujuh kolom bernama?
4. Apa yang terjadi bila `assert` di Sel 3 gagal, dan mengapa itu lebih baik daripada langsung menggambar grafik?
5. Pada Grafik 2, mengapa rata-rata Wake (215,5) berbeda jauh dari mediannya (140,5)? Statistik mana yang Anda sebut ke dosen?
6. Mengapa SC dan ST tidak boleh dicampur tanpa pembeda warna di Grafik 3?
7. Bagaimana temuan N3 di Grafik 4 memengaruhi cara kita membagi data latih dan data uji?
