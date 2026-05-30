"""
Konfigurasi pengembangan lanjutan proyek absensi wajah.

File ini tidak mengganti config.py lama. Tujuannya sebagai konfigurasi tambahan
untuk fitur:
1. Face embedding berbasis ArcFace
2. CNN anti-spoofing foto/video
3. Login guru/admin
4. Enkripsi data sensitif
5. Dashboard web
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Samakan dengan config.py utama bila file ini dipindahkan ke root proyek.
KELAS_AKTIF = "XI TKJ 1"
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
FLIP_HORIZONTAL = True
MIN_DETECTION_CONFIDENCE = 0.70
DETEKSI_BERULANG = 3

# Lokasi file/folder utama.
DATABASE_FILE = BASE_DIR / "database" / "absensi_sekolah.db"
AUTH_DATABASE_FILE = BASE_DIR / "database" / "auth_users.db"
SECRET_KEY_FILE = BASE_DIR / "database" / "secret.key"
DATA_DIR = BASE_DIR / "data_wajah"
MODEL_DIR = BASE_DIR / "model"
EXPORT_DIR = BASE_DIR / "exports"

# Face recognition berbasis embedding.
# Untuk skripsi, gunakan salah satu: ARCFACE atau FACENET.
# Implementasi yang disediakan di paket ini memakai ArcFace melalui InsightFace.
FACE_EMBEDDING_METHOD = "ARCFACE"
#"buffalo_l jika ingin ganti buffalo_sc"
ARCFACE_MODEL_NAME = "buffalo_l"
ARCFACE_EMBEDDING_FILE = MODEL_DIR / "arcface_embeddings.pkl"

# Nilai awal. Wajib dikalibrasi ulang memakai data wajah siswa di kamera/laptop yang sama.
# Similarity makin besar = wajah makin mirip.
ARCFACE_SIMILARITY_THRESHOLD = 0.45

# Anti-spoofing CNN.
USE_CNN_ANTI_SPOOF = False
SPOOF_MODEL_FILE = MODEL_DIR / "spoof_cnn.keras"
SPOOF_LABEL_FILE = MODEL_DIR / "spoof_labels.json"
SPOOF_INPUT_SIZE = 224
SPOOF_REAL_THRESHOLD = 0.70

# Keamanan runtime. Jika False, absensi ditolak bila model anti-spoof belum ada.
ALLOW_ABSEN_IF_SPOOF_MODEL_MISSING = False

# Blink challenge tetap dipakai sebagai active liveness tambahan.
USE_BLINK_CHALLENGE = True
LIVENESS_METHOD = "CNN_ANTI_SPOOF + BLINK_CHALLENGE"
BLINK_EAR_THRESHOLD = 0.30
BLINK_MIN_CLOSED_FRAMES = 1
BLINK_TIMEOUT_SECONDS = 12

# Login dan role.
VALID_ROLES = ("admin", "guru")
SESSION_TIMEOUT_MINUTES = 60

# Dashboard.
DASHBOARD_TITLE = "Dashboard Absensi Face Recognition"
