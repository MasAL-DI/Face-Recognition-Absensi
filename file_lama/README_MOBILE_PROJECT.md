# Pengembangan Proyek: Absensi Siswa Menggunakan HP

File ini mengubah konsep proyek awal dari absensi menggunakan webcam PC menjadi absensi menggunakan HP siswa.

## Konsep

```text
HP siswa
- buka halaman absensi
- ambil selfie wajah
- kirim foto ke server

        ↓ WiFi sekolah

PC lab/server
- menerima foto
- mengenali wajah dengan ArcFace
- mencatat absensi
- menyediakan dashboard sederhana guru/admin
```

## File Baru

| File | Fungsi |
|---|---|
| `mobile_absensi_server.py` | Server Flask untuk absensi menggunakan HP |
| `requirements_mobile.txt` | Dependency tambahan untuk mode HP |

## File Lama yang Tetap Dipakai

| File lama | Status |
|---|---|
| `import_siswa_excel.py` | Tetap dipakai untuk import siswa |
| `daftar_wajah.py` | Bisa tetap dipakai untuk perekaman awal |
| `build_face_embeddings.py` | Tetap dipakai untuk membuat embedding ArcFace |
| `arcface_recognition.py` | Tetap dipakai untuk mengenali wajah |
| `config_lanjutan.py` | Tetap dipakai untuk konfigurasi database, kelas, threshold |
| `auth_db.py` dan `buat_admin.py` | Tetap dipakai untuk login guru/admin |
| `web_dashboard.py` | Tetap bisa dipakai untuk rekap lengkap |

## Cara Memasang

Copy file berikut ke folder utama proyek absensi lama:

```text
mobile_absensi_server.py
requirements_mobile.txt
```

Install dependency tambahan:

```bash
pip install -r requirements_mobile.txt
```

Pastikan dependency lama proyek juga sudah terinstall.

## Urutan Menjalankan

1. Import data siswa:

```bash
python import_siswa_excel.py
```

2. Rekam wajah siswa, bisa pakai cara lama:

```bash
python daftar_wajah.py
```

Atau pakai halaman pendaftaran wajah via HP:

```text
http://IP-PC-LAB:8000/admin/enroll
```

3. Buat embedding ArcFace:

```bash
python build_face_embeddings.py
```

4. Buat akun guru/admin:

```bash
python buat_admin.py
```

5. Jalankan server absensi HP:

```bash
python mobile_absensi_server.py
```

6. Buka halaman admin dari PC/laptop guru:

```text
http://IP-PC-LAB:8000/admin
```

7. Guru membuka sesi absensi.

8. Siswa membuka halaman absensi dari HP yang terhubung ke WiFi sekolah:

```text
http://IP-PC-LAB:8000/absen
```

## Cara Mengetahui IP PC Lab

Di Windows CMD:

```bash
ipconfig
```

Cari bagian WiFi/LAN, misalnya:

```text
IPv4 Address . . . . . . . . . . : 192.168.1.10
```

Berarti link absensi siswa:

```text
http://192.168.1.10:8000/absen
```

## Pengaturan WiFi Sekolah

Di file `mobile_absensi_server.py`, ubah bagian ini sesuai jaringan sekolah:

```python
SCHOOL_NETWORKS = [
    "192.168.1.0/24",
]
```

Contoh:

| IP PC server | SCHOOL_NETWORKS |
|---|---|
| `192.168.1.10` | `"192.168.1.0/24"` |
| `192.168.100.5` | `"192.168.100.0/24"` |
| `10.10.5.20` | `"10.10.0.0/16"` |

Jika masih tahap uji coba dan ingin mematikan pembatasan jaringan sementara:

```python
ENFORCE_SCHOOL_NETWORK = False
```

## Alur Siswa

1. Siswa terhubung ke WiFi sekolah.
2. Siswa membuka link absensi.
3. Siswa mengambil selfie dari kamera depan HP.
4. Siswa menekan tombol kirim absensi.
5. Server mengenali wajah.
6. Jika cocok, status hadir muncul di HP.

## Catatan Penting

- Sistem ini memakai selfie, bukan video real-time.
- Blink liveness dari program desktop lama belum dipakai pada mode HP ini.
- Pengaman utama mode HP adalah WiFi sekolah, sesi aktif guru, dan face recognition.
- Untuk absensi harian, siswa tidak perlu input NIS.
- NIS hanya dipakai saat import data atau pendaftaran wajah.
- Browser HP kadang memerlukan izin kamera. Jika kamera tidak muncul, pilih file/foto dari kamera secara manual.
- Pastikan Windows Firewall mengizinkan akses port 8000.

## Jika HP Tidak Bisa Membuka Server

Cek hal berikut:

1. HP dan PC server harus berada di WiFi/jaringan yang sama.
2. Cek IP PC server dengan `ipconfig`.
3. Pastikan server Flask masih berjalan.
4. Pastikan Windows Firewall tidak memblokir Python.
5. Coba buka dari browser PC sendiri:

```text
http://127.0.0.1:8000/absen
```

6. Coba buka dari HP:

```text
http://IP-PC-LAB:8000/absen
```

## Pengembangan Lanjutan

- QR code link absensi.
- Mode kamera live dengan `getUserMedia`.
- Liveness blink versi web.
- Token sesi dinamis.
- Device logging.
- Export Excel/PDF.
- Integrasi ke dashboard Streamlit lama.
