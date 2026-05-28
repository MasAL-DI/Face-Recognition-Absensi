-- Struktur tambahan untuk pengembangan lanjutan.
-- File ini tidak menghapus tabel lama. Jalankan manual jika ingin mengecek rancangan skema.

CREATE TABLE IF NOT EXISTS users (
    id_user INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin', 'guru')),
    nama_lengkap TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_login TEXT
);

CREATE TABLE IF NOT EXISTS audit_login (
    id_audit INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    role TEXT,
    success INTEGER NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS security_events (
    id_event INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    actor_username TEXT,
    description TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_model_metadata (
    id_model INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,
    model_file TEXT NOT NULL,
    threshold REAL,
    total_identity INTEGER,
    total_sample INTEGER,
    created_at TEXT NOT NULL,
    notes TEXT
);
