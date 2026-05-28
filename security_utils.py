"""
Utilitas keamanan:
- hash password memakai PBKDF2-HMAC-SHA256
- enkripsi/dekripsi field sensitif memakai Fernet

Catatan penting:
SQLite bawaan Python tidak otomatis mengenkripsi seluruh file database.
Yang dilakukan modul ini adalah field-level encryption, yaitu mengenkripsi nilai
tertentu seperti NIS, nama, atau catatan sensitif sebelum disimpan.
Untuk enkripsi satu file database penuh, gunakan SQLCipher. Untuk proyek skripsi,
field-level encryption lebih mudah diuji dan dijelaskan.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

try:
    from config_lanjutan import SECRET_KEY_FILE
except Exception:  # pragma: no cover
    SECRET_KEY_FILE = Path("database/secret.key")

PASSWORD_ITERATIONS = 260_000


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_or_create_fernet_key(key_file: Path = SECRET_KEY_FILE) -> bytes:
    key_file = Path(key_file)
    _ensure_parent(key_file)

    if key_file.exists():
        key = key_file.read_bytes().strip()
        if key:
            return key

    key = Fernet.generate_key()
    key_file.write_bytes(key)
    return key


def get_fernet() -> Fernet:
    return Fernet(load_or_create_fernet_key())


def encrypt_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if text == "":
        return ""
    token = get_fernet().encrypt(text.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if text == "":
        return ""
    try:
        return get_fernet().decrypt(text.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError):
        # Bila data lama belum terenkripsi, kembalikan apa adanya agar migrasi tidak merusak proyek.
        return text


def hash_password(password: str) -> str:
    if not password or len(password) < 6:
        raise ValueError("Password minimal 6 karakter.")

    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii")
    digest_b64 = base64.urlsafe_b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt_b64}${digest_b64}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iter_raw, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iter_raw)
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_b64.encode("ascii"))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False
