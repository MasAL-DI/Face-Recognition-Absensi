"""
Database autentikasi untuk login admin/guru.

Role yang disarankan:
- admin: kelola user, siswa, kelas, ekspor, audit
- guru : membuat sesi absensi, melihat rekap kelas/mapel sendiri
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from security_utils import hash_password, verify_password

try:
    from config_lanjutan import AUTH_DATABASE_FILE, VALID_ROLES
except Exception:  # pragma: no cover
    AUTH_DATABASE_FILE = Path("database/auth_users.db")
    VALID_ROLES = ("admin", "guru")


def get_auth_conn() -> sqlite3.Connection:
    db_path = Path(AUTH_DATABASE_FILE)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db() -> None:
    with get_auth_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id_user INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'guru')),
                nama_lengkap TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_login TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_login (
                id_audit INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                role TEXT,
                success INTEGER NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def create_user(username: str, password: str, role: str, nama_lengkap: str = "") -> int:
    init_auth_db()
    username = username.strip().lower()
    role = role.strip().lower()

    if not username:
        raise ValueError("Username tidak boleh kosong.")
    if role not in VALID_ROLES:
        raise ValueError(f"Role harus salah satu dari: {', '.join(VALID_ROLES)}")

    password_hash = hash_password(password)
    created_at = datetime.now().isoformat(timespec="seconds")

    with get_auth_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO users (username, password_hash, role, nama_lengkap, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username, password_hash, role, nama_lengkap.strip(), created_at),
        )
        conn.commit()
        return int(cur.lastrowid)


def log_login(username: str, role: Optional[str], success: bool, message: str) -> None:
    init_auth_db()
    with get_auth_conn() as conn:
        conn.execute(
            """
            INSERT INTO audit_login (username, role, success, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username, role, int(success), message, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()


def verify_login(username: str, password: str) -> Optional[dict]:
    init_auth_db()
    username = username.strip().lower()

    with get_auth_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? AND is_active = 1",
            (username,),
        ).fetchone()

        if row is None:
            log_login(username, None, False, "Username tidak ditemukan atau nonaktif")
            return None

        ok = verify_password(password, row["password_hash"])
        if not ok:
            log_login(username, row["role"], False, "Password salah")
            return None

        now = datetime.now().isoformat(timespec="seconds")
        conn.execute("UPDATE users SET last_login = ? WHERE id_user = ?", (now, row["id_user"]))
        conn.commit()
        log_login(username, row["role"], True, "Login berhasil")

        return {
            "id_user": int(row["id_user"]),
            "username": row["username"],
            "role": row["role"],
            "nama_lengkap": row["nama_lengkap"] or row["username"],
        }


def list_users() -> list[dict]:
    init_auth_db()
    with get_auth_conn() as conn:
        rows = conn.execute(
            """
            SELECT id_user, username, role, nama_lengkap, is_active, created_at, last_login
            FROM users
            ORDER BY role, username
            """
        ).fetchall()
    return [dict(r) for r in rows]
