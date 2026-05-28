"""
Membuat template embedding ArcFace dari folder data_wajah siswa.

Script ini memakai database.py lama jika tersedia, khususnya fungsi:
- init_db()
- list_siswa(kelas)

Setiap siswa idealnya punya 30-50 foto. Embedding tiap foto dirata-ratakan menjadi
satu template per siswa agar pengenalan lebih stabil.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from arcface_recognition import ArcFaceEngine, l2_normalize

BASE_DIR = Path(__file__).resolve().parent

try:
    from config_lanjutan import ARCFACE_EMBEDDING_FILE, DATA_DIR, KELAS_AKTIF, MODEL_DIR
except Exception:  # pragma: no cover
    ARCFACE_EMBEDDING_FILE = BASE_DIR / "model" / "arcface_embeddings.pkl"
    DATA_DIR = BASE_DIR / "data_wajah"
    MODEL_DIR = BASE_DIR / "model"
    KELAS_AKTIF = "XI TKJ 1"


def _row_to_dict(row) -> dict:
    try:
        return dict(row)
    except Exception:
        return row


def load_students_from_database() -> list[dict]:
    try:
        from database import init_db, list_siswa
    except ImportError as exc:
        raise ImportError("database.py tidak ditemukan. Jalankan script ini di root proyek absensi lama.") from exc

    init_db()
    rows = list_siswa(KELAS_AKTIF)
    return [_row_to_dict(r) for r in rows]


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    arr = np.vstack([l2_normalize(e) for e in embeddings])
    return l2_normalize(np.mean(arr, axis=0))


def resolve_face_folder(row: dict) -> Path:
    folder_wajah = row.get("folder_wajah") or row.get("folder")
    if folder_wajah:
        folder = Path(str(folder_wajah))
        if not folder.is_absolute():
            folder = BASE_DIR / folder
        return folder
    return Path(DATA_DIR) / str(row.get("nis"))


def main() -> None:
    students = load_students_from_database()
    if not students:
        print(f"Tidak ada siswa di kelas {KELAS_AKTIF}. Impor Excel dulu.")
        return

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    engine = ArcFaceEngine(ctx_id=-1)
    templates: list[dict] = []

    print("=" * 80)
    print("BUILD TEMPLATE ARCFACE EMBEDDING")
    print("=" * 80)
    print(f"Kelas: {KELAS_AKTIF}")

    for row in students:
        folder = resolve_face_folder(row)

        if not folder.exists():
            print(f"SKIP {row.get('nama')} - folder tidak ada: {folder}")
            continue

        embeddings = []
        files = sorted(
            list(folder.glob("*.jpg")) +
            list(folder.glob("*.jpeg")) +
            list(folder.glob("*.png"))
        )

        for image_path in files:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  - SKIP file rusak/tidak terbaca: {image_path.name}")
                continue

            emb, _, _ = engine.extract_embedding(img)
            if emb is not None:
                embeddings.append(emb)
            else:
                print(f"  - Wajah tidak terdeteksi: {image_path.name}, shape={img.shape}")

        if not embeddings:
            print(f"SKIP {row.get('nama')} - tidak ada wajah valid di folder")
            continue

        template = {
            "id_siswa": int(row.get("id_siswa")),
            "nis": str(row.get("nis", "")),
            "nama": str(row.get("nama", "")),
            "kelas": str(row.get("kelas", "")),
            "no_absen": int(row.get("no_absen", 0)),
            "folder_wajah": str(folder),
            "jumlah_foto_valid": len(embeddings),
            "embedding": average_embeddings(embeddings),
        }
        templates.append(template)
        print(f"OK {template['no_absen']:02d} | {template['nis']} | {template['nama']} | {len(embeddings)} foto valid")

    payload = {
        "method": "ARCFACE_INSIGHTFACE",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "kelas": KELAS_AKTIF,
        "total_templates": len(templates),
        "templates": templates,
    }

    Path(ARCFACE_EMBEDDING_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(ARCFACE_EMBEDDING_FILE, "wb") as f:
        pickle.dump(payload, f)

    print("=" * 80)
    print(f"Total template: {len(templates)}")
    print(f"Tersimpan: {ARCFACE_EMBEDDING_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
