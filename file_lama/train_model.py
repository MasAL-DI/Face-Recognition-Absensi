"""
train_model.py
Melatih model LBPH dari data wajah siswa.
Wajib dijalankan ulang setelah menambah siswa atau foto wajah baru.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from config import KELAS_AKTIF, MIN_FOTO_DISARANKAN, MODEL_DIR
from database import init_db, list_siswa
from vision_utils import prepare_face_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / MODEL_DIR / "face_model.yml"
LABEL_PATH = BASE_DIR / MODEL_DIR / "label_map.pkl"


def main() -> None:
    init_db()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    siswa_rows = list_siswa(KELAS_AKTIF)
    if not siswa_rows:
        print("Belum ada siswa. Jalankan import_siswa_excel.py dulu.")
        return

    faces: list[np.ndarray] = []
    labels: list[int] = []
    label_map: dict[int, dict[str, object]] = {}
    foto_count: dict[str, int] = {}

    print("=" * 88)
    print("TRAINING MODEL FACE RECOGNITION - LBPH")
    print("=" * 88)
    print(f"Kelas aktif: {KELAS_AKTIF}")
    print(f"Siswa      : {len(siswa_rows)}\n")

    for label_id, siswa in enumerate(siswa_rows):
        folder = BASE_DIR / str(siswa["folder_wajah"])
        label_map[label_id] = {
            "id_siswa": int(siswa["id_siswa"]),
            "no_absen": int(siswa["no_absen"]),
            "nis": str(siswa["nis"]),
            "nama": str(siswa["nama"]),
            "kelas": str(siswa["kelas"]),
            "folder_wajah": str(siswa["folder_wajah"]),
        }

        berhasil = 0
        if folder.exists():
            for file in sorted(folder.glob("*.jpg")):
                face = prepare_face_image(str(file))
                if face is None:
                    continue
                faces.append(face)
                labels.append(label_id)
                berhasil += 1

        key = f"{int(siswa['no_absen']):02d} - {siswa['nis']} - {siswa['nama']}"
        foto_count[key] = berhasil
        status = "OK" if berhasil >= MIN_FOTO_DISARANKAN else "KURANG"
        print(f"{key}: {berhasil} foto [{status}]")

    if not faces:
        print("\nTidak ada foto wajah valid. Training dibatalkan.")
        return

    kurang = [name for name, total in foto_count.items() if total < MIN_FOTO_DISARANKAN]
    if kurang:
        print("\nCatatan: data beberapa siswa masih kurang:")
        for item in kurang:
            print("- " + item)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )

    labels_np = np.array(labels, dtype=np.int32)
    print(f"\nTraining menggunakan {len(faces)} sampel wajah...")
    recognizer.train(faces, labels_np)

    recognizer.save(str(MODEL_PATH))
    with open(LABEL_PATH, "wb") as f:
        pickle.dump(label_map, f)

    scores_per_student: dict[str, list[float]] = defaultdict(list)
    benar = 0
    for face, true_label in zip(faces, labels_np):
        pred_label, distance = recognizer.predict(face)
        true_info = label_map[int(true_label)]
        label = f"{true_info['no_absen']:02d} - {true_info['nama']}"
        scores_per_student[label].append(float(distance))
        if int(pred_label) == int(true_label):
            benar += 1

    print("\nRingkasan skor training")
    print("Skor lebih kecil berarti lebih mirip. Ini bukan persentase akurasi.")
    for label, scores in scores_per_student.items():
        print(
            f"- {label}: "
            f"min={min(scores):.2f}, "
            f"rata-rata={np.mean(scores):.2f}, "
            f"maks={max(scores):.2f}"
        )

    print(f"\nKecocokan pada data training: {benar / len(faces) * 100:.2f}%")
    print("Ini hanya sanity check, bukan hasil pengujian skripsi.")

    print("\n" + "=" * 88)
    print("TRAINING SELESAI")
    print(f"Model    : {MODEL_PATH}")
    print(f"Label map: {LABEL_PATH}")
    print("Langkah berikutnya: python absensi_liveness.py")
    print("=" * 88)


if __name__ == "__main__":
    main()
