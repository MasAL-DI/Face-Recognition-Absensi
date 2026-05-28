"""
database.py
Database SQLite untuk:
- master siswa
- mata pelajaran
- sesi pelajaran
- absensi per sesi
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any

from config import DATABASE_FILE, KELAS_AKTIF

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / DATABASE_FILE


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS siswa (
                id_siswa INTEGER PRIMARY KEY AUTOINCREMENT,
                no_absen INTEGER NOT NULL,
                nis TEXT NOT NULL UNIQUE,
                nama TEXT NOT NULL,
                kelas TEXT NOT NULL,
                folder_wajah TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                UNIQUE(kelas, no_absen)
            );

            CREATE INDEX IF NOT EXISTS idx_siswa_kelas_no_absen
            ON siswa(kelas, no_absen);

            CREATE TABLE IF NOT EXISTS mapel (
                id_mapel INTEGER PRIMARY KEY AUTOINCREMENT,
                nama_mapel TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sesi_pelajaran (
                id_sesi INTEGER PRIMARY KEY AUTOINCREMENT,
                tanggal TEXT NOT NULL,
                kelas TEXT NOT NULL,
                id_mapel INTEGER NOT NULL,
                nama_guru TEXT,
                jam_mulai TEXT NOT NULL,
                jam_selesai TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(id_mapel) REFERENCES mapel(id_mapel),
                UNIQUE(tanggal, kelas, id_mapel, jam_mulai, jam_selesai)
            );

            CREATE INDEX IF NOT EXISTS idx_sesi_tanggal_kelas
            ON sesi_pelajaran(tanggal, kelas);

            CREATE TABLE IF NOT EXISTS absensi (
                id_absensi INTEGER PRIMARY KEY AUTOINCREMENT,
                id_sesi INTEGER NOT NULL,
                id_siswa INTEGER NOT NULL,
                jam_absen TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'HADIR',
                skor_lbph REAL,
                liveness_method TEXT,
                liveness_passed INTEGER NOT NULL DEFAULT 0,
                liveness_detail TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(id_sesi) REFERENCES sesi_pelajaran(id_sesi) ON DELETE CASCADE,
                FOREIGN KEY(id_siswa) REFERENCES siswa(id_siswa) ON DELETE CASCADE,
                UNIQUE(id_sesi, id_siswa)
            );

            CREATE INDEX IF NOT EXISTS idx_absensi_sesi
            ON absensi(id_sesi);

            CREATE INDEX IF NOT EXISTS idx_absensi_siswa
            ON absensi(id_siswa);

            CREATE TABLE IF NOT EXISTS log_uji (
                id_log INTEGER PRIMARY KEY AUTOINCREMENT,
                tanggal_waktu TEXT NOT NULL,
                jenis_uji TEXT NOT NULL,
                nama_subjek TEXT,
                kelas TEXT,
                kondisi_cahaya TEXT,
                jarak_cm INTEGER,
                expected_label TEXT,
                predicted_label TEXT,
                skor_lbph REAL,
                liveness_passed INTEGER,
                hasil TEXT,
                catatan TEXT
            );
            """
        )


def make_folder_name(no_absen: int, nis: str, nama: str) -> str:
    import re
    slug = normalize_text(nama).lower()
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"[^a-z0-9_]+", "", slug)
    if not slug:
        slug = "siswa"
    return f"data_wajah/{int(no_absen):02d}_{normalize_text(nis)}_{slug}"


def upsert_siswa(no_absen: int, nis: str, nama: str, kelas: str = KELAS_AKTIF) -> tuple[str, int, str]:
    """
    Return: (status, id_siswa, folder_wajah)
    status: inserted / updated
    """
    init_db()
    no_absen = int(no_absen)
    nis = normalize_text(nis)
    nama = normalize_text(nama)
    kelas = normalize_text(kelas) or KELAS_AKTIF

    if no_absen <= 0:
        raise ValueError("Nomor absen harus lebih besar dari 0.")
    if not nis or not nama:
        raise ValueError("NIS dan nama tidak boleh kosong.")

    with get_connection() as conn:
        existing = conn.execute(
            "SELECT * FROM siswa WHERE nis = ?",
            (nis,),
        ).fetchone()

        if existing:
            conn.execute(
                """
                UPDATE siswa
                SET no_absen = ?, nama = ?, kelas = ?, updated_at = ?
                WHERE id_siswa = ?
                """,
                (no_absen, nama, kelas, now_str(), int(existing["id_siswa"])),
            )
            return "updated", int(existing["id_siswa"]), str(existing["folder_wajah"])

        folder_wajah = make_folder_name(no_absen, nis, nama)
        cur = conn.execute(
            """
            INSERT INTO siswa(no_absen, nis, nama, kelas, folder_wajah, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (no_absen, nis, nama, kelas, folder_wajah, now_str()),
        )
        return "inserted", int(cur.lastrowid), folder_wajah


def list_siswa(kelas: str | None = KELAS_AKTIF) -> list[sqlite3.Row]:
    init_db()
    with get_connection() as conn:
        if kelas:
            return conn.execute(
                """
                SELECT * FROM siswa
                WHERE kelas = ?
                ORDER BY no_absen ASC, nama ASC
                """,
                (normalize_text(kelas),),
            ).fetchall()

        return conn.execute(
            """
            SELECT * FROM siswa
            ORDER BY kelas ASC, no_absen ASC, nama ASC
            """
        ).fetchall()


def get_siswa_by_no_absen(no_absen: int, kelas: str = KELAS_AKTIF) -> sqlite3.Row | None:
    init_db()
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT * FROM siswa
            WHERE no_absen = ? AND kelas = ?
            """,
            (int(no_absen), normalize_text(kelas)),
        ).fetchone()


def get_siswa_by_nis(nis: str) -> sqlite3.Row | None:
    init_db()
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM siswa WHERE nis = ?",
            (normalize_text(nis),),
        ).fetchone()


def list_kelas() -> list[str]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT kelas FROM siswa ORDER BY kelas ASC"
        ).fetchall()
    return [str(row["kelas"]) for row in rows]


def get_or_create_mapel(nama_mapel: str) -> int:
    init_db()
    nama_mapel = normalize_text(nama_mapel)
    if not nama_mapel:
        raise ValueError("Nama mata pelajaran tidak boleh kosong.")

    with get_connection() as conn:
        row = conn.execute(
            "SELECT id_mapel FROM mapel WHERE nama_mapel = ?",
            (nama_mapel,),
        ).fetchone()
        if row:
            return int(row["id_mapel"])

        cur = conn.execute(
            "INSERT INTO mapel(nama_mapel, created_at) VALUES (?, ?)",
            (nama_mapel, now_str()),
        )
        return int(cur.lastrowid)


def create_or_get_sesi(
    tanggal: str,
    kelas: str,
    nama_mapel: str,
    jam_mulai: str,
    jam_selesai: str,
    nama_guru: str = "",
) -> sqlite3.Row:
    init_db()
    tanggal = normalize_text(tanggal)
    kelas = normalize_text(kelas)
    jam_mulai = normalize_text(jam_mulai)
    jam_selesai = normalize_text(jam_selesai)
    nama_guru = normalize_text(nama_guru)
    id_mapel = get_or_create_mapel(nama_mapel)

    with get_connection() as conn:
        existing = conn.execute(
            """
            SELECT sp.*, m.nama_mapel
            FROM sesi_pelajaran sp
            JOIN mapel m ON m.id_mapel = sp.id_mapel
            WHERE sp.tanggal = ?
              AND sp.kelas = ?
              AND sp.id_mapel = ?
              AND sp.jam_mulai = ?
              AND sp.jam_selesai = ?
            """,
            (tanggal, kelas, id_mapel, jam_mulai, jam_selesai),
        ).fetchone()

        if existing:
            return existing

        cur = conn.execute(
            """
            INSERT INTO sesi_pelajaran(
                tanggal, kelas, id_mapel, nama_guru, jam_mulai, jam_selesai, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (tanggal, kelas, id_mapel, nama_guru, jam_mulai, jam_selesai, now_str()),
        )
        id_sesi = int(cur.lastrowid)

        return conn.execute(
            """
            SELECT sp.*, m.nama_mapel
            FROM sesi_pelajaran sp
            JOIN mapel m ON m.id_mapel = sp.id_mapel
            WHERE sp.id_sesi = ?
            """,
            (id_sesi,),
        ).fetchone()


def get_sesi_by_id(id_sesi: int) -> sqlite3.Row | None:
    init_db()
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT sp.*, m.nama_mapel
            FROM sesi_pelajaran sp
            JOIN mapel m ON m.id_mapel = sp.id_mapel
            WHERE sp.id_sesi = ?
            """,
            (int(id_sesi),),
        ).fetchone()


def list_sesi(limit: int | None = 30) -> list[sqlite3.Row]:
    init_db()
    params: list[Any] = []
    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT ?"
        params.append(int(limit))

    with get_connection() as conn:
        return conn.execute(
            f"""
            SELECT
                sp.*,
                m.nama_mapel,
                COUNT(a.id_absensi) AS jumlah_hadir
            FROM sesi_pelajaran sp
            JOIN mapel m ON m.id_mapel = sp.id_mapel
            LEFT JOIN absensi a ON a.id_sesi = sp.id_sesi
            GROUP BY sp.id_sesi
            ORDER BY sp.tanggal DESC, sp.jam_mulai DESC, sp.id_sesi DESC
            {limit_sql}
            """,
            tuple(params),
        ).fetchall()


def get_hadir_siswa_ids(id_sesi: int) -> set[int]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id_siswa FROM absensi WHERE id_sesi = ?",
            (int(id_sesi),),
        ).fetchall()
    return {int(row["id_siswa"]) for row in rows}


def record_absensi(
    id_sesi: int,
    id_siswa: int,
    jam_absen: str,
    skor_lbph: float | None,
    liveness_method: str,
    liveness_passed: bool,
    liveness_detail: str = "",
) -> bool:
    init_db()
    with get_connection() as conn:
        try:
            conn.execute(
                """
                INSERT INTO absensi(
                    id_sesi, id_siswa, jam_absen, status,
                    skor_lbph, liveness_method, liveness_passed, liveness_detail, created_at
                )
                VALUES (?, ?, ?, 'HADIR', ?, ?, ?, ?, ?)
                """,
                (
                    int(id_sesi),
                    int(id_siswa),
                    jam_absen,
                    skor_lbph,
                    liveness_method,
                    1 if liveness_passed else 0,
                    normalize_text(liveness_detail),
                    now_str(),
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def get_rekap_sesi(id_sesi: int) -> list[sqlite3.Row]:
    init_db()
    sesi = get_sesi_by_id(id_sesi)
    if sesi is None:
        return []

    with get_connection() as conn:
        return conn.execute(
            """
            SELECT
                s.id_siswa,
                s.no_absen,
                s.nis,
                s.nama,
                s.kelas,
                a.jam_absen,
                a.status,
                a.skor_lbph,
                a.liveness_method,
                a.liveness_passed,
                a.liveness_detail
            FROM siswa s
            LEFT JOIN absensi a
              ON a.id_siswa = s.id_siswa
             AND a.id_sesi = ?
            WHERE s.kelas = ?
            ORDER BY s.no_absen ASC, s.nama ASC
            """,
            (int(id_sesi), str(sesi["kelas"])),
        ).fetchall()


def get_ringkasan_tanggal(tanggal: str) -> list[sqlite3.Row]:
    init_db()
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT
                sp.id_sesi,
                sp.tanggal,
                sp.kelas,
                m.nama_mapel,
                sp.jam_mulai,
                sp.jam_selesai,
                sp.nama_guru,
                COUNT(DISTINCT a.id_siswa) AS jumlah_hadir,
                (
                    SELECT COUNT(*)
                    FROM siswa s
                    WHERE s.kelas = sp.kelas
                ) AS total_siswa
            FROM sesi_pelajaran sp
            JOIN mapel m ON m.id_mapel = sp.id_mapel
            LEFT JOIN absensi a ON a.id_sesi = sp.id_sesi
            WHERE sp.tanggal = ?
            GROUP BY sp.id_sesi
            ORDER BY sp.jam_mulai ASC, m.nama_mapel ASC
            """,
            (normalize_text(tanggal),),
        ).fetchall()


def search_siswa(keyword: str) -> list[sqlite3.Row]:
    init_db()
    keyword = normalize_text(keyword)
    pattern = f"%{keyword}%"

    with get_connection() as conn:
        return conn.execute(
            """
            SELECT * FROM siswa
            WHERE CAST(no_absen AS TEXT) LIKE ?
               OR nis LIKE ?
               OR nama LIKE ?
               OR kelas LIKE ?
            ORDER BY kelas ASC, no_absen ASC, nama ASC
            """,
            (pattern, pattern, pattern, pattern),
        ).fetchall()


def get_absensi_by_siswa(id_siswa: int) -> list[sqlite3.Row]:
    init_db()
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT
                a.*,
                sp.tanggal,
                sp.kelas,
                sp.jam_mulai,
                sp.jam_selesai,
                sp.nama_guru,
                m.nama_mapel
            FROM absensi a
            JOIN sesi_pelajaran sp ON sp.id_sesi = a.id_sesi
            JOIN mapel m ON m.id_mapel = sp.id_mapel
            WHERE a.id_siswa = ?
            ORDER BY sp.tanggal DESC, sp.jam_mulai DESC
            """,
            (int(id_siswa),),
        ).fetchall()


def delete_absensi_all() -> None:
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM absensi")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'absensi'")


def delete_sesi_and_absensi_all() -> None:
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM absensi")
        conn.execute("DELETE FROM sesi_pelajaran")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'absensi'")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'sesi_pelajaran'")


def log_uji(
    jenis_uji: str,
    nama_subjek: str = "",
    kelas: str = "",
    kondisi_cahaya: str = "",
    jarak_cm: int | None = None,
    expected_label: str = "",
    predicted_label: str = "",
    skor_lbph: float | None = None,
    liveness_passed: bool | None = None,
    hasil: str = "",
    catatan: str = "",
) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO log_uji(
                tanggal_waktu, jenis_uji, nama_subjek, kelas,
                kondisi_cahaya, jarak_cm, expected_label, predicted_label,
                skor_lbph, liveness_passed, hasil, catatan
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_str(),
                normalize_text(jenis_uji),
                normalize_text(nama_subjek),
                normalize_text(kelas),
                normalize_text(kondisi_cahaya),
                jarak_cm,
                normalize_text(expected_label),
                normalize_text(predicted_label),
                skor_lbph,
                None if liveness_passed is None else (1 if liveness_passed else 0),
                normalize_text(hasil),
                normalize_text(catatan),
            ),
        )
