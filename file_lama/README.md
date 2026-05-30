# Sistem Absensi Siswa Berbasis Face Recognition

Proyek ini adalah prototipe sistem absensi siswa per mata pelajaran berbasis **Face Recognition** dengan dukungan **liveness detection** sederhana. Sistem dirancang untuk digunakan di lingkungan SMK/sekolah dengan komputer lokal sebagai pusat pemrosesan, kamera sebagai input wajah, database sebagai penyimpanan data, serta dashboard web untuk melihat rekap absensi.

Proyek ini dapat dijalankan sebagai sistem absensi lokal menggunakan webcam/laptop/PC sekolah. Versi lanjutan mendukung ArcFace embedding, blink challenge, login admin/guru, dashboard Streamlit, dan opsi CNN anti-spoofing.

---

## 1. Gambaran Umum Sistem

Alur kerja sistem:

```text
Import data siswa dari Excel
        ↓
Rekam data wajah siswa
        ↓
Bangun template/embedding wajah
        ↓
Mulai sesi absensi
        ↓
Kamera mendeteksi wajah siswa
        ↓
Sistem mengenali wajah
        ↓
Validasi liveness/kedipan
        ↓
Absensi dicatat ke database
        ↓
Guru/admin melihat rekap di dashboard
```

---

## 2. Fitur Utama

### 2.1 Import Data Siswa dari Excel

Sistem mendukung import data siswa dari file `.xlsx`.

Format minimal kolom Excel:

| Kolom | Keterangan |
|---|---|
| `no_absen` | Nomor absen siswa |
| `nis` | Nomor induk siswa |
| `nama` | Nama siswa |
| `kelas` | Kelas siswa, opsional jika memakai kelas default |

Jika kolom kelas kosong atau tidak tersedia, sistem menggunakan kelas default dari konfigurasi.

---

### 2.2 Perekaman Data Wajah

Sistem menyediakan modul perekaman wajah siswa berdasarkan data yang sudah diimpor dari Excel.

Rekomendasi jumlah foto:

| Kebutuhan | Jumlah |
|---|---:|
| Minimum disarankan | 30 foto/siswa |
| Ideal | 30–50 foto/siswa |
| Target default | 50 foto/siswa |

Instruksi perekaman:

- Wajah siswa menghadap kamera.
- Cahaya ruangan cukup.
- Wajah tidak tertutup masker/topi berlebihan.
- Posisi wajah dibuat sedikit bervariasi.
- Tekan `SPACE` untuk menyimpan foto.
- Tekan `ESC` untuk keluar dari mode perekaman.

---

### 2.3 Face Recognition Berbasis ArcFace

Versi lanjutan menggunakan **ArcFace embedding** melalui InsightFace. Sistem membandingkan embedding wajah dari kamera dengan template wajah siswa yang sudah dibuat.

Prinsip kerja:

```text
Foto wajah siswa → ekstraksi embedding → template wajah
Wajah dari kamera → ekstraksi embedding → cosine similarity
Similarity di atas threshold → siswa dikenali
```

Nilai similarity semakin besar berarti wajah semakin mirip. Threshold perlu dikalibrasi berdasarkan kamera dan kondisi pencahayaan di lokasi uji coba.

---

### 2.4 Liveness Detection Blink Challenge

Sistem menggunakan liveness sederhana berbasis kedipan mata.

Alur:

```text
Wajah dikenali
        ↓
Sistem meminta siswa berkedip
        ↓
MediaPipe FaceMesh membaca landmark mata
        ↓
Sistem menghitung Eye Aspect Ratio/EAR
        ↓
Jika mata tertutup lalu terbuka kembali, liveness valid
```

---

### 2.5 CNN Anti-Spoofing Opsional

Sistem memiliki modul CNN anti-spoofing untuk membedakan wajah asli dan spoof seperti foto HP, foto cetak, atau video replay.

Struktur dataset:

```text
dataset_spoof/
  train/
    real/
    spoof/
  val/
    real/
    spoof/
```

Catatan: fitur ini opsional. Jika dataset masih sedikit, hasil CNN anti-spoofing dapat kurang stabil. Untuk versi awal proyek magang, blink challenge dapat dijadikan fitur utama, sedangkan CNN anti-spoofing dapat dijadikan pengembangan lanjutan.

---

### 2.6 Dashboard Web

Dashboard berbasis Streamlit digunakan untuk:

- Login admin/guru.
- Melihat ringkasan jumlah siswa, kelas, sesi, dan data hadir.
- Melihat tabel rekap absensi.
- Memfilter data berdasarkan tanggal, kelas, dan mata pelajaran.
- Mengekspor data absensi ke CSV.
- Melihat daftar user dan audit login sederhana.

---

## 3. Struktur File Proyek

```text
project_absensi/
│
├── absensi_lanjutan_arcface_cnn.py
├── anti_spoof_cnn.py
├── arcface_recognition.py
├── auth_db.py
├── buat_admin.py
├── build_face_embeddings.py
├── config.py
├── config_lanjutan.py
├── daftar_wajah.py
├── import_siswa_excel.py
├── log_uji_manual.py
├── rekap.py
├── reset_data.py
├── security_utils.py
├── train_spoof_cnn.py
├── vision_utils.py
├── web_dashboard.py
│
├── database/
├── data_wajah/
├── model/
├── exports/
└── data_siswa.xlsx
```

Keterangan file:

| File | Fungsi |
|---|---|
| `config.py` | Konfigurasi utama versi awal |
| `config_lanjutan.py` | Konfigurasi fitur lanjutan |
| `import_siswa_excel.py` | Import data siswa dari Excel |
| `daftar_wajah.py` | Perekaman foto wajah siswa |
| `build_face_embeddings.py` | Membuat template embedding ArcFace |
| `arcface_recognition.py` | Modul identifikasi wajah berbasis ArcFace |
| `vision_utils.py` | Utilitas kamera, deteksi wajah, dan blink liveness |
| `absensi_lanjutan_arcface_cnn.py` | Program utama absensi lanjutan |
| `anti_spoof_cnn.py` | Inference CNN anti-spoofing |
| `train_spoof_cnn.py` | Training CNN anti-spoofing |
| `auth_db.py` | Database login admin/guru |
| `buat_admin.py` | Membuat akun admin/guru |
| `web_dashboard.py` | Dashboard web Streamlit |
| `rekap.py` | Rekap absensi lewat terminal |
| `log_uji_manual.py` | Mencatat hasil uji manual |
| `reset_data.py` | Reset data absensi/sesi |
| `security_utils.py` | Hash password dan enkripsi data sensitif |

---

## 4. Kebutuhan Sistem

### 4.1 Perangkat Keras Minimum

| Komponen | Minimum |
|---|---|
| Komputer | Laptop/PC sekolah |
| CPU | Intel i3 atau setara |
| RAM | 4 GB |
| Kamera | Webcam 720p |
| Storage | Minimal 10 GB kosong |
| OS | Windows/Linux |

### 4.2 Perangkat Keras Disarankan

| Komponen | Rekomendasi |
|---|---|
| Komputer | Intel i5 / Intel N100 / setara |
| RAM | 8 GB |
| Kamera | Webcam 1080p |
| Storage | SSD |
| Kondisi ruangan | Cahaya cukup, tidak backlight |

### 4.3 Perangkat Lunak

- Python 3.10 atau 3.11
- OpenCV
- MediaPipe
- NumPy
- Pandas
- OpenPyXL
- Streamlit
- InsightFace
- ONNXRuntime
- TensorFlow/Keras, jika memakai CNN anti-spoofing
- Cryptography

---

## 5. Instalasi

### 5.1 Buat Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 5.2 Install Dependency

Dependency utama:

```bash
pip install opencv-python mediapipe numpy pandas openpyxl streamlit cryptography
pip install insightface onnxruntime
```

Jika memakai CNN anti-spoofing:

```bash
pip install tensorflow
```

---

## 6. Konfigurasi

Konfigurasi utama ada di `config.py` dan `config_lanjutan.py`.

Contoh konfigurasi:

```python
KELAS_AKTIF = "XI TKJ 1"
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
FLIP_HORIZONTAL = True
```

Konfigurasi ArcFace:

```python
FACE_EMBEDDING_METHOD = "ARCFACE"
ARCFACE_MODEL_NAME = "buffalo_sc"
ARCFACE_SIMILARITY_THRESHOLD = 0.45
```

Catatan: nilai `ARCFACE_SIMILARITY_THRESHOLD` harus diuji ulang sesuai data wajah, kamera, dan pencahayaan di sekolah.

---

## 7. Cara Menjalankan

### 7.1 Siapkan Data Siswa

Buat file:

```text
data_siswa.xlsx
```

Contoh isi:

| no_absen | nis | nama | kelas |
|---:|---|---|---|
| 1 | 12345 | Ahmad Fajar | XI TKJ 1 |
| 2 | 12346 | Budi Santoso | XI TKJ 1 |

Jalankan:

```bash
python import_siswa_excel.py
```

---

### 7.2 Rekam Wajah Siswa

```bash
python daftar_wajah.py
```

Pilih mode:

```text
[1] Rekam urut dari siswa pertama yang belum mencapai target
[2] Rekam berdasarkan nomor absen
[3] Keluar
```

---

### 7.3 Buat Embedding ArcFace

```bash
python build_face_embeddings.py
```

Output:

```text
model/arcface_embeddings.pkl
```

---

### 7.4 Buat Akun Admin/Guru

```bash
python buat_admin.py
```

Masukkan username, nama lengkap, role, dan password.

---

### 7.5 Jalankan Program Absensi

```bash
python absensi_lanjutan_arcface_cnn.py
```

Masukkan data sesi:

- Tanggal
- Mata pelajaran
- Nama guru
- Jam mulai
- Jam selesai

Setelah itu kamera akan aktif dan sistem siap mencatat absensi.

---

### 7.6 Jalankan Dashboard

```bash
streamlit run web_dashboard.py
```

Dashboard dapat dibuka melalui browser.

---

## 8. Urutan Cepat

```bash
python import_siswa_excel.py
python daftar_wajah.py
python build_face_embeddings.py
python buat_admin.py
python absensi_lanjutan_arcface_cnn.py
streamlit run web_dashboard.py
```

Jika memakai anti-spoofing CNN:

```bash
python train_spoof_cnn.py
```

---

## 9. Pengujian Sistem

Skenario uji yang disarankan:

| No | Jenis Uji | Tujuan |
|---:|---|---|
| 1 | Wajah terdaftar | Memastikan siswa dikenali |
| 2 | Wajah tidak terdaftar | Memastikan orang lain ditolak |
| 3 | Foto dari HP | Menguji spoofing |
| 4 | Cahaya terang | Menguji kondisi ideal |
| 5 | Cahaya redup | Menguji kondisi kurang cahaya |
| 6 | Backlight | Menguji cahaya belakang |
| 7 | Jarak dekat/jauh | Menentukan jarak optimal |
| 8 | Wajah tertutup | Menguji kondisi wajah tidak penuh |
| 9 | Siswa sudah absen | Mencegah pencatatan ganda |
| 10 | Liveness kedip | Memastikan blink challenge berjalan |

Untuk mencatat hasil uji manual:

```bash
python log_uji_manual.py
```

---

## 10. Kalibrasi Threshold ArcFace

Langkah sederhana:

1. Uji beberapa siswa yang sudah terdaftar.
2. Catat similarity wajah benar.
3. Uji orang yang tidak terdaftar.
4. Catat similarity wajah salah.
5. Pilih threshold yang paling aman.

Interpretasi:

| Kondisi | Keterangan |
|---|---|
| Threshold terlalu rendah | Orang lain bisa dikenali sebagai siswa |
| Threshold terlalu tinggi | Siswa asli bisa gagal dikenali |
| Threshold ideal | Siswa asli diterima, orang lain ditolak |

---

## 11. Keamanan dan Privasi

Karena sistem menyimpan NIS, nama, wajah, dan data absensi, hal berikut perlu diperhatikan:

- Dashboard hanya boleh diakses admin/guru.
- Folder data wajah tidak boleh disebarkan.
- Gunakan password admin/guru yang kuat.
- Simpan database di komputer sekolah yang aman.
- Jelaskan bahwa data wajah hanya digunakan untuk kebutuhan absensi.
- Hapus data uji jika proyek selesai atau jika diminta sekolah.

---

## 12. Reset Data

Untuk menghapus data uji:

```bash
python reset_data.py
```

Pilihan:

```text
[1] Hapus semua data absensi saja
[2] Hapus semua sesi pelajaran + absensi
[3] Batal
```

---

## 13. Troubleshooting

### Kamera tidak terbuka

Cek:

```python
CAMERA_INDEX = 0
```

Jika memakai webcam eksternal, coba:

```python
CAMERA_INDEX = 1
```

### Wajah tidak terdeteksi

Solusi:

- Pastikan cahaya cukup.
- Hindari backlight.
- Dekatkan wajah ke kamera.
- Pastikan kamera tidak blur.
- Hindari wajah tertutup.

### File embedding belum ada

Jalankan:

```bash
python build_face_embeddings.py
```

### Template siswa kosong

Kemungkinan penyebab:

- Data siswa belum diimpor.
- Folder wajah siswa kosong.
- Foto wajah rusak/tidak terbaca.
- Wajah pada foto tidak terdeteksi.

Solusi:

```bash
python import_siswa_excel.py
python daftar_wajah.py
python build_face_embeddings.py
```

### Dashboard kosong

Kemungkinan penyebab:

- Database belum dibuat.
- Belum ada data absensi.
- Program absensi belum dijalankan.
- Struktur tabel database belum sesuai.

Solusi:

```bash
python absensi_lanjutan_arcface_cnn.py
streamlit run web_dashboard.py
```

---

## 14. Batasan Sistem

- Akurasi dipengaruhi kualitas kamera dan pencahayaan.
- Wajah tertutup dapat menurunkan akurasi.
- Blink challenge masih sederhana.
- CNN anti-spoofing membutuhkan dataset yang cukup.
- Sistem awal masih berbasis komputer lokal.
- Threshold ArcFace perlu dikalibrasi ulang.

---

## 15. Rencana Pengembangan

- Membuat absensi berbasis web/mobile.
- Membuat mode absensi memakai HP siswa.
- Membuat terminal IoT kelas.
- Menggunakan database MySQL/PostgreSQL.
- Menambahkan QR sesi.
- Menambahkan export Excel/PDF.
- Menambahkan audit perubahan absensi.
- Menambahkan role wali kelas, guru mapel, dan admin sekolah.
- Menambahkan deployment server lokal di jaringan sekolah.

---

## 16. Rekomendasi Judul

1. **Sistem Absensi Siswa Berbasis Face Recognition Menggunakan ArcFace dan Blink Liveness Detection pada SMK**
2. **Smart Attendance System Berbasis Pengenalan Wajah dan Liveness Detection untuk Lingkungan Sekolah**
3. **Rancang Bangun Sistem Absensi Siswa Per Mata Pelajaran Berbasis Face Recognition dan Dashboard Web**
4. **Prototipe Sistem Absensi Siswa Berbasis ArcFace dengan Validasi Kedipan Mata dan Rekap Dashboard**

---

## 17. Status Proyek

```text
Prototipe pengembangan untuk proyek magang/skripsi.
```

Sistem dapat digunakan untuk demonstrasi dan pengujian terbatas, tetapi masih memerlukan kalibrasi, pengujian lapangan, dan penyesuaian keamanan sebelum digunakan sebagai sistem absensi resmi sekolah.

---

## 18. Lisensi dan Penggunaan

Proyek ini dibuat untuk pembelajaran, magang, penelitian, dan pengembangan sistem absensi sekolah. Penggunaan data wajah siswa harus mendapatkan izin dari pihak sekolah dan mengikuti aturan privasi yang berlaku.
