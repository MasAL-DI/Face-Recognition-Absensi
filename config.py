"""
config.py
Konfigurasi utama proyek skripsi:
Sistem Absensi Siswa Per Mata Pelajaran Berbasis Face Recognition
dengan Liveness Detection sederhana berbasis kedipan mata.
"""

# Ubah sesuai kelas tempat uji coba.
KELAS_AKTIF = "XI TKJ 1"

# Data wajah
TARGET_FOTO = 50
MIN_FOTO_DISARANKAN = 30

# Kamera
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# Jika tampilan kamera terasa mirror, ubah ke False.
FLIP_HORIZONTAL = True

# LBPH:
# skor lebih kecil = wajah lebih mirip.
# 55 dipilih sebagai nilai awal dari hasil uji awal pengguna.
MATCH_THRESHOLD = 55.0

# Wajah harus dikenali stabil beberapa frame sebelum absensi dicatat.
DETEKSI_BERULANG = 3

# Face detection
MIN_DETECTION_CONFIDENCE = 0.70

# Liveness detection
# Mode saat ini: blink challenge.
# Siswa harus berkedip setelah wajah dikenali.
USE_LIVENESS = True
LIVENESS_METHOD = "BLINK_CHALLENGE"

# EAR atau Eye Aspect Ratio.
# Nilai bisa berbeda tergantung kamera dan jarak.
BLINK_EAR_THRESHOLD = 0.30
BLINK_MIN_CLOSED_FRAMES = 1
BLINK_TIMEOUT_SECONDS = 12

# File dan folder
DATABASE_FILE = "database/absensi_sekolah.db"
DATA_DIR = "data_wajah"
MODEL_DIR = "model"
EXPORT_DIR = "exports"
