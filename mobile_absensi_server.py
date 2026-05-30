"""
mobile_absensi_server.py

Server absensi berbasis HP siswa.

Konsep:
- PC lab menjalankan file ini sebagai server lokal.
- Siswa membuka halaman /absen dari HP yang terhubung ke WiFi sekolah.
- HP mengambil selfie, lalu foto dikirim ke server.
- Server memproses face recognition memakai ArcFace embedding yang sudah dibuat.
- Jika wajah cocok dan sesi aktif, absensi dicatat ke database.

Jalankan:
    python mobile_absensi_server.py

Buka dari HP:
    http://IP-PC-LAB:8000/absen

Buka dashboard admin sederhana:
    http://IP-PC-LAB:8000/admin
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
from datetime import datetime, timedelta
from functools import wraps
from io import BytesIO
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template_string, request, session, url_for

try:
    from PIL import Image, ImageOps
except Exception:  # Pillow disarankan, tetapi server tetap bisa jalan tanpa Pillow.
    Image = None
    ImageOps = None

from arcface_recognition import ArcFaceRecognizer
from config_lanjutan import (
    ARCFACE_SIMILARITY_THRESHOLD,
    DATA_DIR,
    DATABASE_FILE,
    KELAS_AKTIF,
    LIVENESS_METHOD,
)
from security_utils import decrypt_text

try:
    from auth_db import verify_login
except Exception:
    verify_login = None


BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "database" / "mobile_active_session.json"

# Ganti sesuai jaringan sekolah.
# Contoh:
#   WiFi sekolah 192.168.1.x  -> "192.168.1.0/24"
#   WiFi sekolah 192.168.100.x -> "192.168.100.0/24"
#
# Untuk uji coba awal, private networks umum diizinkan.
SCHOOL_NETWORKS = [
    "127.0.0.0/8",
    "192.168.0.0/16",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.18.74",
    "72.27.36.137"

]

# Jika True, request dari luar SCHOOL_NETWORKS ditolak.
ENFORCE_SCHOOL_NETWORK = True

# Durasi default sesi absensi aktif setelah guru membuka sesi.
DEFAULT_SESSION_DURATION_MINUTES = 15

# Agar foto upload dari HP tidak terlalu besar.
MAX_IMAGE_WIDTH = 1280

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ganti-secret-key-ini-untuk-produksi")

_RECOGNIZER: Optional[ArcFaceRecognizer] = None


def db_connect() -> sqlite3.Connection:
    db_path = Path(DATABASE_FILE)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_minimal_schema() -> None:
    """
    Membuat tabel minimal jika database belum ada.
    Jika project lama sudah punya database.py sendiri, fungsi ini tidak merusak tabel lama.
    """
    with db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mapel (
                id_mapel INTEGER PRIMARY KEY AUTOINCREMENT,
                nama_mapel TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sesi_pelajaran (
                id_sesi INTEGER PRIMARY KEY AUTOINCREMENT,
                tanggal TEXT NOT NULL,
                kelas TEXT NOT NULL,
                id_mapel INTEGER NOT NULL,
                nama_guru TEXT,
                jam_mulai TEXT NOT NULL,
                jam_selesai TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(id_mapel) REFERENCES mapel(id_mapel)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS absensi (
                id_absensi INTEGER PRIMARY KEY AUTOINCREMENT,
                id_sesi INTEGER NOT NULL,
                id_siswa INTEGER NOT NULL,
                jam_absen TEXT NOT NULL,
                skor_lbph REAL,
                liveness_method TEXT,
                liveness_passed INTEGER,
                liveness_detail TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(id_sesi, id_siswa)
            )
            """
        )
        conn.commit()


def get_lan_ip() -> str:
    """
    Mengambil IP LAN PC server untuk ditampilkan ke guru.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def client_ip() -> str:
    """
    Mengambil IP client. Untuk jaringan lokal biasa, request.remote_addr cukup.
    """
    forwarded = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    return forwarded or request.remote_addr or ""


def is_allowed_network(ip_raw: str) -> bool:
    if not ENFORCE_SCHOOL_NETWORK:
        return True

    try:
        ip = ip_address(ip_raw)
        networks = [ip_network(net) for net in SCHOOL_NETWORKS]
        return any(ip in net for net in networks)
    except Exception:
        return False


def network_required_json(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ip = client_ip()
        if not is_allowed_network(ip):
            return jsonify(
                {
                    "ok": False,
                    "message": "Akses ditolak. Hubungkan HP ke WiFi sekolah.",
                    "client_ip": ip,
                }
            ), 403
        return fn(*args, **kwargs)

    return wrapper


def network_required_page(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ip = client_ip()
        if not is_allowed_network(ip):
            return render_template_string(ERROR_PAGE, title="Akses ditolak", message=f"HP harus terhubung ke WiFi sekolah. IP terdeteksi: {ip}"), 403
        return fn(*args, **kwargs)

    return wrapper


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("admin_login"))
        return fn(*args, **kwargs)

    return wrapper


def get_recognizer() -> ArcFaceRecognizer:
    """
    Lazy load ArcFace recognizer agar server tidak berat saat start.
    Pastikan sudah menjalankan:
        python build_face_embeddings.py
    """
    global _RECOGNIZER
    if _RECOGNIZER is None:
        _RECOGNIZER = ArcFaceRecognizer(threshold=ARCFACE_SIMILARITY_THRESHOLD, ctx_id=-1)
    return _RECOGNIZER


def row_to_public_dict(row: sqlite3.Row | dict) -> dict:
    d = dict(row)
    for key in ["nis", "nama", "nama_guru"]:
        if key in d and d[key] is not None:
            d[key] = decrypt_text(str(d[key]))
    return d


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]


def get_or_create_mapel(conn: sqlite3.Connection, nama_mapel: str) -> int:
    nama_mapel = " ".join(nama_mapel.strip().split())
    if not nama_mapel:
        raise ValueError("Nama mata pelajaran tidak boleh kosong.")

    row = conn.execute(
        "SELECT id_mapel FROM mapel WHERE nama_mapel = ?",
        (nama_mapel,),
    ).fetchone()

    if row:
        return int(row["id_mapel"])

    columns = get_table_columns(conn, "mapel")
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "created_at" in columns:
        cur = conn.execute(
            "INSERT INTO mapel (nama_mapel, created_at) VALUES (?, ?)",
            (nama_mapel, now_text),
        )
    else:
        cur = conn.execute(
            "INSERT INTO mapel (nama_mapel) VALUES (?)",
            (nama_mapel,),
        )

    return int(cur.lastrowid)


def create_sesi(
    tanggal: str,
    kelas: str,
    nama_mapel: str,
    jam_mulai: str,
    jam_selesai: str,
    nama_guru: str = "",
) -> int:
    ensure_minimal_schema()

    tanggal = " ".join(tanggal.strip().split())
    kelas = " ".join(kelas.strip().split())
    nama_mapel = " ".join(nama_mapel.strip().split())
    jam_mulai = " ".join(jam_mulai.strip().split())
    jam_selesai = " ".join(jam_selesai.strip().split())
    nama_guru = " ".join(nama_guru.strip().split())

    if not tanggal:
        raise ValueError("Tanggal tidak boleh kosong.")
    if not kelas:
        raise ValueError("Kelas tidak boleh kosong.")
    if not nama_mapel:
        raise ValueError("Mata pelajaran tidak boleh kosong.")
    if not jam_mulai:
        raise ValueError("Jam mulai tidak boleh kosong.")
    if not jam_selesai:
        raise ValueError("Jam selesai tidak boleh kosong.")

    with db_connect() as conn:
        id_mapel = get_or_create_mapel(conn, nama_mapel)

        # Jika sesi yang sama sudah ada, pakai sesi lama agar tidak error UNIQUE.
        existing = conn.execute(
            """
            SELECT id_sesi
            FROM sesi_pelajaran
            WHERE tanggal = ?
              AND kelas = ?
              AND id_mapel = ?
              AND jam_mulai = ?
              AND jam_selesai = ?
            """,
            (tanggal, kelas, id_mapel, jam_mulai, jam_selesai),
        ).fetchone()

        if existing:
            return int(existing["id_sesi"])

        columns = get_table_columns(conn, "sesi_pelajaran")
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data = {
            "tanggal": tanggal,
            "kelas": kelas,
            "id_mapel": id_mapel,
            "nama_guru": nama_guru,
            "jam_mulai": jam_mulai,
            "jam_selesai": jam_selesai,
            "created_at": now_text,
        }

        insert_cols = [c for c in data.keys() if c in columns]
        placeholders = ",".join(["?"] * len(insert_cols))

        sql = f"""
            INSERT INTO sesi_pelajaran ({",".join(insert_cols)})
            VALUES ({placeholders})
        """

        cur = conn.execute(sql, tuple(data[c] for c in insert_cols))
        conn.commit()

        return int(cur.lastrowid)

def get_sesi(id_sesi: int) -> Optional[dict]:
    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT sp.*, m.nama_mapel
            FROM sesi_pelajaran sp
            JOIN mapel m ON m.id_mapel = sp.id_mapel
            WHERE sp.id_sesi = ?
            """,
            (id_sesi,),
        ).fetchone()
    return row_to_public_dict(row) if row else None


def set_active_session(id_sesi: int, duration_minutes: int) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    expires_at = datetime.now() + timedelta(minutes=duration_minutes)
    STATE_FILE.write_text(
        json.dumps(
            {
                "id_sesi": int(id_sesi),
                "expires_at": expires_at.isoformat(timespec="seconds"),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def clear_active_session() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def get_active_session() -> Optional[dict]:
    if not STATE_FILE.exists():
        return None

    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        expires_at = datetime.fromisoformat(state["expires_at"])
        if datetime.now() > expires_at:
            clear_active_session()
            return None

        sesi = get_sesi(int(state["id_sesi"]))
        if not sesi:
            clear_active_session()
            return None

        sesi["expires_at"] = state["expires_at"]
        return sesi
    except Exception:
        clear_active_session()
        return None


def list_students(kelas: str = KELAS_AKTIF) -> list[dict]:
    with db_connect() as conn:
        if not table_exists(conn, "siswa"):
            return []
        rows = conn.execute(
            """
            SELECT id_siswa, no_absen, nis, nama, kelas, folder_wajah
            FROM siswa
            WHERE kelas = ?
            ORDER BY no_absen ASC
            """,
            (kelas,),
        ).fetchall()
    return [row_to_public_dict(r) for r in rows]


def get_student_by_id(id_siswa: int) -> Optional[dict]:
    with db_connect() as conn:
        if not table_exists(conn, "siswa"):
            return None
        row = conn.execute(
            """
            SELECT id_siswa, no_absen, nis, nama, kelas, folder_wajah
            FROM siswa
            WHERE id_siswa = ?
            """,
            (id_siswa,),
        ).fetchone()
    return row_to_public_dict(row) if row else None


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def get_absen_count(id_sesi: int) -> int:
    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT id_siswa) AS n
            FROM absensi
            WHERE id_sesi = ?
              AND jam_absen IS NOT NULL
              AND TRIM(jam_absen) != ''
            """,
            (int(id_sesi),),
        ).fetchone()

    return int(row["n"]) if row else 0


def already_attended(id_sesi: int, id_siswa: int) -> bool:
    with db_connect() as conn:
        row = conn.execute(
            "SELECT id_absensi FROM absensi WHERE id_sesi = ? AND id_siswa = ?",
            (id_sesi, id_siswa),
        ).fetchone()
    return row is not None


def insert_attendance(id_sesi: int, id_siswa: int, similarity: float, detail: str) -> bool:
    if already_attended(id_sesi, id_siswa):
        return False

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    jam_absen = datetime.now().strftime("%H:%M:%S")

    with db_connect() as conn:
        columns = [
            r["name"]
            for r in conn.execute("PRAGMA table_info(absensi)").fetchall()
        ]

        data = {
            "id_sesi": int(id_sesi),
            "id_siswa": int(id_siswa),
            "jam_absen": jam_absen,
            "status": "HADIR",
            "skor_lbph": float(similarity),
            "liveness_method": "MOBILE_LIVE_CAMERA_FACE_RECOGNITION",
            "liveness_passed": 1,
            "liveness_detail": detail,
            "created_at": now_text,
        }

        insert_cols = [col for col in data if col in columns]

        if not insert_cols:
            raise RuntimeError("Kolom tabel absensi tidak ditemukan.")

        placeholders = ",".join(["?"] * len(insert_cols))
        sql = f"""
            INSERT INTO absensi ({",".join(insert_cols)})
            VALUES ({placeholders})
        """

        conn.execute(sql, tuple(data[col] for col in insert_cols))
        conn.commit()

    return True

def next_photo_index(folder: Path) -> int:
    numbers = []
    for file in folder.glob("foto_*.jpg"):
        try:
            numbers.append(int(file.stem.split("_")[1]))
        except Exception:
            pass
    return max(numbers, default=-1) + 1


def read_uploaded_image(file_storage) -> np.ndarray:
    """
    Membaca foto upload dari HP menjadi BGR image untuk OpenCV/ArcFace.
    Menggunakan Pillow jika tersedia agar rotasi EXIF selfie HP bisa diperbaiki.
    """
    data = file_storage.read()

    if Image is not None and ImageOps is not None:
        img = Image.open(BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.array(img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is None or bgr.size == 0:
        raise ValueError("File foto tidak valid atau tidak terbaca.")

    h, w = bgr.shape[:2]
    if w > MAX_IMAGE_WIDTH:
        scale = MAX_IMAGE_WIDTH / float(w)
        bgr = cv2.resize(bgr, (MAX_IMAGE_WIDTH, int(h * scale)))

    return bgr


BASE_CSS = """
<style>
:root {
    --bg: #0f172a;
    --card: #111827;
    --card2: #1f2937;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --success: #22c55e;
    --danger: #ef4444;
    --warning: #f59e0b;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: linear-gradient(135deg, #020617, #0f172a);
    color: var(--text);
}
.container {
    width: min(960px, 94vw);
    margin: 0 auto;
    padding: 24px 0 48px;
}
.card {
    background: rgba(17, 24, 39, 0.95);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 20px;
    padding: 22px;
    margin: 16px 0;
    box-shadow: 0 18px 50px rgba(0,0,0,.25);
}
h1, h2, h3 { margin-top: 0; }
p { color: var(--muted); line-height: 1.55; }
a { color: var(--accent); text-decoration: none; }
input, select, button {
    width: 100%;
    padding: 13px 14px;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.28);
    background: #020617;
    color: var(--text);
    margin: 7px 0 14px;
    font-size: 15px;
}
button {
    background: linear-gradient(135deg, #0284c7, #0369a1);
    border: 0;
    cursor: pointer;
    font-weight: 700;
}
button.secondary { background: #334155; }
button.danger { background: #b91c1c; }
.grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
}
.metric {
    background: #020617;
    border-radius: 16px;
    padding: 16px;
    border: 1px solid rgba(148, 163, 184, 0.16);
}
.metric b { display: block; font-size: 28px; margin-top: 6px; }
.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: #1e293b;
    color: #e2e8f0;
    font-size: 13px;
    margin: 3px 4px 3px 0;
}
.ok { color: var(--success); }
.bad { color: var(--danger); }
.warn { color: var(--warning); }
.preview {
    display: none;
    width: 100%;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    margin: 10px 0 14px;
}
.result {
    padding: 16px;
    border-radius: 16px;
    margin-top: 14px;
    background: #020617;
    border: 1px solid rgba(148, 163, 184, 0.2);
    white-space: pre-wrap;
}
table {
    width: 100%;
    border-collapse: collapse;
    overflow: hidden;
    border-radius: 14px;
}
td, th {
    padding: 10px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.16);
    text-align: left;
}
@media (max-width: 680px) {
    .grid { grid-template-columns: 1fr; }
    .container { width: 92vw; padding-top: 14px; }
    .card { padding: 18px; border-radius: 18px; }
}
</style>
"""


ERROR_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
""" + BASE_CSS + """
</head>
<body>
<div class="container">
  <div class="card">
    <h1>{{ title }}</h1>
    <p>{{ message }}</p>
    <a href="/absen">Kembali</a>
  </div>
</div>
</body>
</html>
"""


ABSEN_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Absensi Siswa Live</title>
""" + BASE_CSS + """
<style>
.camera-box {
    position: relative;
    width: 100%;
    background: #020617;
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.25);
    margin-top: 14px;
}
#video {
    width: 100%;
    display: block;
    transform: scaleX(-1);
}
#canvas {
    display: none;
}
.camera-overlay {
    position: absolute;
    left: 50%;
    top: 50%;
    width: 68%;
    aspect-ratio: 1 / 1.2;
    transform: translate(-50%, -50%);
    border: 3px solid rgba(56, 189, 248, 0.95);
    border-radius: 50%;
    box-shadow: 0 0 0 999px rgba(0, 0, 0, 0.22);
    pointer-events: none;
}
.status-live {
    padding: 12px 14px;
    border-radius: 14px;
    margin-top: 12px;
    background: #020617;
    border: 1px solid rgba(148, 163, 184, 0.2);
    white-space: pre-wrap;
    line-height: 1.45;
}
.small-note {
    font-size: 13px;
    color: var(--muted);
}
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>Smart Absensi Live</h1>
    <p>Hubungkan HP ke WiFi sekolah, izinkan kamera, lalu arahkan wajah ke area panduan. Sistem akan membaca wajah secara otomatis tanpa upload manual.</p>

    {% if active %}
      <span class="badge ok">Sesi aktif</span>
      <span class="badge">{{ active.tanggal }}</span>
      <span class="badge">{{ active.kelas }}</span>
      <span class="badge">{{ active.nama_mapel }}</span>
      <span class="badge">Berakhir: {{ active.expires_at }}</span>
    {% else %}
      <span class="badge bad">Belum ada sesi aktif</span>
    {% endif %}

    <div class="camera-box">
      <video id="video" autoplay playsinline muted></video>
      <div class="camera-overlay"></div>
      <canvas id="canvas"></canvas>
    </div>

    <button id="startBtn" type="button">Mulai Kamera Live</button>
    <button id="stopBtn" class="danger" type="button" disabled>Stop Kamera</button>

    <div id="result" class="status-live">Status: Kamera belum aktif.</div>
    <p class="small-note">
      Catatan: jika kamera tidak muncul di HP, buka memakai Chrome dan pastikan izin kamera diberikan.
      Pada beberapa HP/browser, kamera hanya mau aktif pada HTTPS atau localhost.
    </p>
  </div>
</div>

<script>
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

let stream = null;
let scanTimer = null;
let isProcessing = false;
let attendanceDone = false;

const ACTIVE_SESSION = {{ "true" if active else "false" }};

function setResult(text) {
  result.textContent = text;
}

async function startCamera() {
  if (!ACTIVE_SESSION) {
    setResult("GAGAL\\nBelum ada sesi absensi aktif. Minta guru membuka sesi terlebih dahulu.");
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 720 },
        height: { ideal: 960 }
      },
      audio: false
    });

    video.srcObject = stream;

    startBtn.disabled = true;
    stopBtn.disabled = false;
    attendanceDone = false;

    setResult("Status: Kamera aktif. Arahkan wajah ke area panduan. Sistem sedang memindai...");

    scanTimer = setInterval(captureAndSendFrame, 1600);
  } catch (err) {
    setResult(
      "GAGAL MEMBUKA KAMERA\\n" +
      "Pastikan izin kamera diizinkan di browser.\\n\\n" +
      "Detail: " + err
    );
  }
}

function stopCamera() {
  if (scanTimer) {
    clearInterval(scanTimer);
    scanTimer = null;
  }

  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  isProcessing = false;

  if (!attendanceDone) {
    setResult("Status: Kamera dihentikan.");
  }
}

async function captureAndSendFrame() {
  if (isProcessing || attendanceDone) return;
  if (!video.videoWidth || !video.videoHeight) return;

  isProcessing = true;

  const maxWidth = 720;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);

  const ctx = canvas.getContext("2d");

  /*
    Karena tampilan video di-mirror dengan CSS, frame yang dikirim juga dibuat mirror
    agar sesuai dengan tampilan siswa di layar HP.
  */
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  canvas.toBlob(async (blob) => {
    if (!blob) {
      isProcessing = false;
      return;
    }

    const data = new FormData();
    data.append("foto", blob, "live_frame.jpg");

    try {
      setResult("Status: Memindai wajah live...");

      const res = await fetch("/api/absen", {
        method: "POST",
        body: data
      });

      const json = await res.json();

      if (json.ok) {
        attendanceDone = true;
        setResult(
          "ABSENSI BERHASIL\\n" +
          "Nama: " + json.nama + "\\n" +
          "Kelas: " + json.kelas + "\\n" +
          "Jam: " + json.jam_absen + "\\n" +
          "Similarity: " + json.similarity
        );
        stopCamera();
        return;
      }

      if (json.message && json.message.includes("sudah tercatat hadir")) {
        attendanceDone = true;
        setResult(
          "SUDAH ABSEN\\n" +
          json.message
        );
        stopCamera();
        return;
      }

      setResult(
        "Status: Kamera aktif.\\n" +
        "Hasil terakhir: " + (json.message || "Wajah belum dikenali.") + "\\n" +
        "Tetap arahkan wajah ke kamera."
      );
    } catch (err) {
      setResult("Error koneksi ke server: " + err);
    } finally {
      isProcessing = false;
    }
  }, "image/jpeg", 0.85);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

window.addEventListener("beforeunload", stopCamera);
</script>
</body>
</html>
"""

ADMIN_LOGIN_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login Admin</title>
""" + BASE_CSS + """
</head>
<body>
<div class="container">
  <div class="card">
    <h1>Login Guru/Admin</h1>
    <p>Gunakan akun yang dibuat melalui <code>python buat_admin.py</code>.</p>
    {% if error %}<p class="bad">{{ error }}</p>{% endif %}
    <form method="post">
      <label>Username</label>
      <input name="username" required>
      <label>Password</label>
      <input name="password" type="password" required>
      <button type="submit">Masuk</button>
    </form>
  </div>
</div>
</body>
</html>
"""


ADMIN_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admin Absensi Mobile</title>
""" + BASE_CSS + """
</head>
<body>
<div class="container">
  <div class="card">
    <h1>Dashboard Sesi Absensi Mobile</h1>
    <p>Masuk sebagai: <b>{{ user.nama_lengkap }}</b> / {{ user.role }}</p>
    <span class="badge">Server IP: {{ lan_ip }}</span>
    <span class="badge">Link siswa: http://{{ lan_ip }}:8000/absen</span>
    <span class="badge">Client IP kamu: {{ ip }}</span>
    <p><a href="/admin/enroll">Pendaftaran wajah via HP</a> · <a href="/absen">Halaman siswa</a> · <a href="/admin/logout">Logout</a></p>
  </div>

  <div class="grid">
    <div class="metric">
      Status sesi
      {% if active %}
        <b class="ok">Aktif</b>
        <p>{{ active.tanggal }} · {{ active.kelas }} · {{ active.nama_mapel }}<br>Berakhir: {{ active.expires_at }}</p>
      {% else %}
        <b class="bad">Tidak aktif</b>
        <p>Buka sesi baru agar siswa dapat absen.</p>
      {% endif %}
    </div>
    <div class="metric">
        Jumlah hadir sesi aktif
        <b id="hadirCount">{{ hadir }}</b>
        <p id="hadirInfo">Data dihitung dari tabel absensi.</p>
    </div>
  </div>

  <div class="card">
    <h2>Buka Sesi Absensi</h2>
    <form method="post" action="/admin/session/start">
      <label>Tanggal</label>
      <input name="tanggal" value="{{ today }}" required>
      <label>Kelas</label>
      <input name="kelas" value="{{ kelas_aktif }}" required>
      <label>Mata pelajaran</label>
      <input name="nama_mapel" placeholder="Contoh: Matematika" required>
      <label>Nama guru</label>
      <input name="nama_guru" placeholder="Opsional">
      <div class="grid">
        <div>
          <label>Jam mulai</label>
          <input name="jam_mulai" value="{{ now_hhmm }}" required>
        </div>
        <div>
          <label>Jam selesai</label>
          <input name="jam_selesai" value="{{ end_hhmm }}" required>
        </div>
      </div>
      <label>Durasi sesi aktif untuk absensi HP, menit</label>
      <input name="duration" type="number" value="15" min="1" max="180" required>
      <button type="submit">Buka Sesi</button>
    </form>

    <form method="post" action="/admin/session/stop">
      <button class="danger" type="submit">Tutup Sesi Aktif</button>
    </form>
  </div>
</div>
<script>
async function updateHadirCount() {
  try {
    const res = await fetch("/api/admin/status", {
      cache: "no-store"
    });

    const data = await res.json();

    const hadirCount = document.getElementById("hadirCount");
    const hadirInfo = document.getElementById("hadirInfo");

    if (!hadirCount || !hadirInfo) return;

    if (data.ok && data.active) {
      hadirCount.textContent = data.hadir;
      hadirInfo.textContent =
        "Sesi: " + data.tanggal + " · " +
        data.kelas + " · " +
        data.nama_mapel + " · update otomatis";
    } else {
      hadirCount.textContent = "0";
      hadirInfo.textContent = "Tidak ada sesi aktif.";
    }
  } catch (err) {
    const hadirInfo = document.getElementById("hadirInfo");
    if (hadirInfo) {
      hadirInfo.textContent = "Gagal update jumlah hadir: " + err;
    }
  }
}

updateHadirCount();
setInterval(updateHadirCount, 2000);
</script>
</body>
</html>
"""


ENROLL_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pendaftaran Wajah via HP</title>
""" + BASE_CSS + """
</head>
<body>
<div class="container">
  <div class="card">
    <h1>Pendaftaran Wajah via HP</h1>
    <p>Pilih siswa, ambil beberapa foto wajah dengan kamera depan HP, lalu upload. Setelah selesai, jalankan <code>python build_face_embeddings.py</code> di PC server.</p>
    <p><a href="/admin">Kembali dashboard</a></p>
  </div>

  <div class="card">
    <form method="post" action="/admin/enroll/upload" enctype="multipart/form-data">
      <label>Pilih siswa</label>
      <select name="id_siswa" required>
        {% for s in students %}
          <option value="{{ s.id_siswa }}">{{ "%02d"|format(s.no_absen|int) }} · {{ s.nis }} · {{ s.nama }} · {{ s.kelas }}</option>
        {% endfor %}
      </select>

      <label>Ambil/upload foto wajah</label>
      <input name="foto" type="file" accept="image/*" capture="user" multiple required>

      <button type="submit">Upload Foto Wajah</button>
    </form>
  </div>

  <div class="card">
    <h2>Tips pengambilan foto</h2>
    <p>Ambil 20–30 foto per siswa jika memungkinkan. Variasikan wajah lurus, agak kiri, agak kanan, jarak dekat, dan jarak sedang. Hindari backlight dan wajah tertutup.</p>
  </div>
</div>
</body>
</html>
"""


@app.route("/")
def index():
    return redirect(url_for("absen_page"))


@app.route("/absen")
@network_required_page
def absen_page():
    return render_template_string(ABSEN_PAGE, active=get_active_session())


@app.route("/api/absen", methods=["POST"])
@network_required_json
def api_absen():
    try:
        active = get_active_session()
        if not active:
            return jsonify({
                "ok": False,
                "message": "Belum ada sesi absensi aktif. Minta guru membuka sesi."
            }), 400

        if "foto" not in request.files:
            return jsonify({
                "ok": False,
                "message": "File foto tidak ditemukan."
            }), 400

        frame = read_uploaded_image(request.files["foto"])
        recognizer = get_recognizer()
        match = recognizer.identify(frame)

        if not match.matched or not match.info:
            return jsonify({
                "ok": False,
                "message": f"Wajah tidak dikenali. {match.message}",
                "similarity": round(float(match.similarity), 3),
            }), 200

        info = match.info
        id_siswa = int(info["id_siswa"])

        siswa = get_student_by_id(id_siswa)
        if not siswa:
            return jsonify({
                "ok": False,
                "message": "Wajah dikenali, tetapi data siswa tidak ditemukan di database."
            }), 400

        if str(siswa["kelas"]).strip() != str(active["kelas"]).strip():
            return jsonify({
                "ok": False,
                "message": (
                    f"Siswa dikenali sebagai {siswa['nama']}, "
                    f"tetapi kelasnya {siswa['kelas']}, bukan sesi {active['kelas']}."
                ),
            }), 400

        id_sesi = int(active["id_sesi"])

        if already_attended(id_sesi, id_siswa):
            return jsonify({
                "ok": False,
                "message": f"{siswa['nama']} sudah tercatat hadir pada sesi ini.",
                "nama": siswa["nama"],
                "kelas": siswa["kelas"],
            }), 200

        detail = (
            f"Mobile live camera; "
            f"ArcFace similarity={float(match.similarity):.3f}; "
            f"IP={client_ip()}"
        )

        insert_attendance(
            id_sesi=id_sesi,
            id_siswa=id_siswa,
            similarity=float(match.similarity),
            detail=detail,
        )

        return jsonify({
            "ok": True,
            "message": "Absensi berhasil.",
            "id_siswa": id_siswa,
            "nama": siswa["nama"],
            "nis": siswa["nis"],
            "kelas": siswa["kelas"],
            "jam_absen": datetime.now().strftime("%H:%M:%S"),
            "similarity": round(float(match.similarity), 3),
        })

    except FileNotFoundError:
        return jsonify({
            "ok": False,
            "message": "File embedding belum ada. Jalankan python build_face_embeddings.py"
        }), 500

    except Exception as exc:
        return jsonify({
            "ok": False,
            "message": f"Gagal memproses absensi: {type(exc).__name__}: {exc}"
        }), 500
    
@app.errorhandler(404)
def handle_404(exc):
    if request.path.startswith("/api/"):
        return jsonify({
            "ok": False,
            "message": f"Endpoint tidak ditemukan: {request.path}"
        }), 404

    return render_template_string(
        ERROR_PAGE,
        title="Halaman tidak ditemukan",
        message=f"Path tidak ditemukan: {request.path}",
    ), 404


@app.errorhandler(500)
def handle_500(exc):
    if request.path.startswith("/api/"):
        return jsonify({
            "ok": False,
            "message": "Terjadi error internal server. Lihat terminal Python untuk detail."
        }), 500

    return render_template_string(
        ERROR_PAGE,
        title="Internal Server Error",
        message="Terjadi error internal server. Lihat terminal Python untuk detail.",
    ), 500

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if verify_login is None:
            error = "auth_db.py tidak bisa digunakan. Pastikan file auth_db.py tersedia."
        else:
            user = verify_login(username, password)
            if user:
                session["user"] = user
                return redirect(url_for("admin_page"))
            error = "Login gagal. Periksa username/password."

    return render_template_string(ADMIN_LOGIN_PAGE, error=error)


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


@app.route("/admin")
@login_required
def admin_page():
    active = get_active_session()
    hadir = get_absen_count(int(active["id_sesi"])) if active else 0
    now = datetime.now()
    return render_template_string(
        ADMIN_PAGE,
        user=session["user"],
        active=active,
        hadir=hadir,
        lan_ip=get_lan_ip(),
        ip=client_ip(),
        today=now.strftime("%Y-%m-%d"),
        now_hhmm=now.strftime("%H:%M"),
        end_hhmm=(now + timedelta(minutes=45)).strftime("%H:%M"),
        kelas_aktif=KELAS_AKTIF,
    )

@app.route("/api/admin/status")
@login_required
def api_admin_status():
    active = get_active_session()

    if not active:
        return jsonify({
            "ok": True,
            "active": False,
            "hadir": 0,
            "message": "Tidak ada sesi aktif."
        })

    hadir = get_absen_count(int(active["id_sesi"]))

    return jsonify({
        "ok": True,
        "active": True,
        "id_sesi": int(active["id_sesi"]),
        "tanggal": active["tanggal"],
        "kelas": active["kelas"],
        "nama_mapel": active["nama_mapel"],
        "expires_at": active["expires_at"],
        "hadir": hadir,
    })


@app.route("/admin/session/start", methods=["POST"])
@login_required
def admin_session_start():
    tanggal = request.form.get("tanggal", "").strip()
    kelas = request.form.get("kelas", "").strip()
    nama_mapel = request.form.get("nama_mapel", "").strip()
    nama_guru = request.form.get("nama_guru", "").strip()
    jam_mulai = request.form.get("jam_mulai", "").strip()
    jam_selesai = request.form.get("jam_selesai", "").strip()
    duration = int(request.form.get("duration", DEFAULT_SESSION_DURATION_MINUTES))

    id_sesi = create_sesi(tanggal, kelas, nama_mapel, jam_mulai, jam_selesai, nama_guru)
    set_active_session(id_sesi, duration)
    return redirect(url_for("admin_page"))


@app.route("/admin/session/stop", methods=["POST"])
@login_required
def admin_session_stop():
    clear_active_session()
    return redirect(url_for("admin_page"))


@app.route("/admin/enroll")
@login_required
@network_required_page
def admin_enroll():
    students = list_students(KELAS_AKTIF)
    return render_template_string(ENROLL_PAGE, students=students)


@app.route("/admin/enroll/upload", methods=["POST"])
@login_required
@network_required_page
def admin_enroll_upload():
    id_siswa_raw = request.form.get("id_siswa", "")
    if not id_siswa_raw.isdigit():
        return render_template_string(ERROR_PAGE, title="Gagal", message="ID siswa tidak valid."), 400

    siswa = get_student_by_id(int(id_siswa_raw))
    if not siswa:
        return render_template_string(ERROR_PAGE, title="Gagal", message="Siswa tidak ditemukan."), 404

    files = request.files.getlist("foto")
    if not files:
        return render_template_string(ERROR_PAGE, title="Gagal", message="Tidak ada foto yang diupload."), 400

    folder_raw = siswa.get("folder_wajah") or str(Path(DATA_DIR) / str(siswa["nis"]))
    folder = Path(folder_raw)
    if not folder.is_absolute():
        folder = BASE_DIR / folder
    folder.mkdir(parents=True, exist_ok=True)

    saved = 0
    idx = next_photo_index(folder)
    for fs in files:
        try:
            frame = read_uploaded_image(fs)
            out = folder / f"foto_{idx:03d}.jpg"
            cv2.imwrite(str(out), frame)
            idx += 1
            saved += 1
        except Exception:
            continue

    return render_template_string(
        ERROR_PAGE,
        title="Upload selesai",
        message=f"Berhasil menyimpan {saved} foto untuk {siswa['nama']}. Setelah semua foto terkumpul, jalankan python build_face_embeddings.py di PC server.",
    )


def print_startup_info() -> None:
    ip = get_lan_ip()
    print("=" * 80)
    print("SERVER ABSENSI HP SISWA")
    print("=" * 80)
    print(f"Halaman siswa : http://{ip}:8000/absen")
    print(f"Admin/guru    : http://{ip}:8000/admin")
    print(f"Kelas aktif   : {KELAS_AKTIF}")
    print(f"Database      : {DATABASE_FILE}")
    print("Pastikan HP siswa terhubung ke WiFi yang satu jaringan dengan PC server.")
    print("=" * 80)


if __name__ == "__main__":
    ensure_minimal_schema()
    print_startup_info()
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False,
        ssl_context="adhoc"
    )
