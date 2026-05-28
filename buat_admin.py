"""CLI sederhana untuk membuat akun admin/guru."""

from __future__ import annotations

import getpass
import sqlite3

from auth_db import create_user, init_auth_db, list_users


def main() -> None:
    init_auth_db()
    print("=" * 70)
    print("BUAT AKUN LOGIN ADMIN/GURU")
    print("=" * 70)
    print("User saat ini:")
    for user in list_users():
        status = "aktif" if user["is_active"] else "nonaktif"
        print(f"- {user['username']} | {user['role']} | {status}")

    username = input("Username: ").strip().lower()
    nama = input("Nama lengkap: ").strip()
    role = input("Role [admin/guru]: ").strip().lower() or "guru"
    password = getpass.getpass("Password minimal 6 karakter: ")
    password2 = getpass.getpass("Ulangi password: ")

    if password != password2:
        print("Password tidak sama.")
        return

    try:
        user_id = create_user(username, password, role, nama)
        print(f"Akun berhasil dibuat. ID user: {user_id}")
    except sqlite3.IntegrityError:
        print("Username sudah ada. Gunakan username lain.")
    except Exception as exc:
        print(f"Gagal membuat akun: {exc}")


if __name__ == "__main__":
    main()
