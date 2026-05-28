"""
daftar_wajah.py
Perekaman wajah siswa dari daftar yang sudah diimpor dari Excel.
Versi ini menyimpan frame wajah berwarna penuh agar kompatibel dengan ArcFace/InsightFace.

Catatan:
- Untuk LBPH lama, crop grayscale 200x200 masih bisa dipakai melalui train_model.py lama.
- Untuk ArcFace, data yang lebih aman adalah frame/crop berwarna karena InsightFace akan mendeteksi wajah dan mengambil embedding sendiri.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import mediapipe as mp

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    KELAS_AKTIF,
    MIN_DETECTION_CONFIDENCE,
    MIN_FOTO_DISARANKAN,
    TARGET_FOTO,
)
from database import get_siswa_by_no_absen, init_db, list_siswa
from vision_utils import detection_to_bbox, largest_detection, maybe_flip

BASE_DIR = Path(__file__).resolve().parent


def count_photos(folder: Path) -> int:
    return len(list(folder.glob("*.jpg")))


def next_photo_index(folder: Path) -> int:
    indices = []
    for file in folder.glob("foto_*.jpg"):
        m = re.match(r"foto_(\d+)\.jpg$", file.name.lower())
        if m:
            indices.append(int(m.group(1)))
    return max(indices, default=-1) + 1


def show_roster(rows) -> None:
    print("\nDaftar siswa:")
    print("-" * 88)
    print(f"{'No':<6} {'NIS':<16} {'Nama':<32} {'Kelas':<16} {'Foto'}")
    print("-" * 88)
    for row in rows:
        folder = BASE_DIR / str(row["folder_wajah"])
        total = count_photos(folder)
        status = "OK" if total >= MIN_FOTO_DISARANKAN else "KURANG"
        print(
            f"{int(row['no_absen']):<6} "
            f"{str(row['nis']):<16} "
            f"{str(row['nama']):<32} "
            f"{str(row['kelas']):<16} "
            f"{total:>3}/{TARGET_FOTO:<3} {status}"
        )
    print("-" * 88)


def capture_student(row) -> None:
    no_absen = int(row["no_absen"])
    nis = str(row["nis"])
    nama = str(row["nama"])
    kelas = str(row["kelas"])
    folder = BASE_DIR / str(row["folder_wajah"])
    folder.mkdir(parents=True, exist_ok=True)

    photo_idx = next_photo_index(folder)

    print("\n" + "=" * 88)
    print("PEREKAMAN WAJAH")
    print("=" * 88)
    print(f"No. Absen : {no_absen:02d}")
    print(f"NIS       : {nis}")
    print(f"Nama      : {nama}")
    print(f"Kelas     : {kelas}")
    print(f"Foto awal : {count_photos(folder)}/{TARGET_FOTO}")
    print("\nInstruksi:")
    print("- Siswa berdiri/duduk di depan kamera.")
    print("- Wajah tidak tertutup, cahaya cukup.")
    print("- Variasikan sedikit arah kepala.")
    print("- SPACE untuk simpan foto berwarna.")
    print("- ESC untuk berhenti pada siswa ini.")
    print("=" * 88)

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka.")
        face_detection.close()
        return

    last_face = None
    last_bbox = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca kamera.")
                break

            frame = maybe_flip(frame)
            frame = cv2.resize(frame, (960, 640))
            display = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)

            last_face = None
            last_bbox = None

            if results.detections:
                det = largest_detection(results.detections, frame.shape)
                if det and float(det.score[0]) >= MIN_DETECTION_CONFIDENCE:
                    bbox = detection_to_bbox(det, frame.shape)

                    # Simpan frame berwarna penuh agar ArcFace/InsightFace dapat
                    # melakukan deteksi dan ekstraksi embedding sendiri.
                    last_face = frame.copy()
                    last_bbox = bbox

                    cv2.rectangle(
                        display,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 220, 0),
                        2,
                    )

            total = count_photos(folder)
            cv2.rectangle(display, (0, 0), (960, 92), (24, 24, 36), -1)
            cv2.putText(
                display,
                f"{no_absen:02d} | {nis} | {nama}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.76,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                f"Kelas {kelas} | Foto {total}/{TARGET_FOTO} | SPACE simpan | ESC berhenti",
                (16, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (190, 220, 255),
                2,
            )

            if last_face is None:
                status = "Arahkan wajah dengan jelas ke kamera"
                color = (0, 0, 220)
            else:
                status = "Wajah terdeteksi - tekan SPACE"
                color = (0, 220, 0)

            cv2.rectangle(display, (0, 600), (960, 640), (24, 24, 36), -1)
            cv2.putText(display, status, (16, 626), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)

            progress = min(1.0, total / max(TARGET_FOTO, 1))
            cv2.rectangle(display, (16, 574), (456, 590), (70, 70, 70), -1)
            cv2.rectangle(display, (16, 574), (16 + int(440 * progress), 590), (0, 220, 0), -1)

            cv2.imshow("Perekaman Wajah Siswa", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key == 32 and last_face is not None:
                filename = folder / f"foto_{photo_idx:03d}.jpg"
                ok = cv2.imwrite(str(filename), last_face)
                if ok:
                    print(f"Tersimpan: {filename.name}")
                    photo_idx += 1

                if count_photos(folder) >= TARGET_FOTO:
                    print(f"Target {TARGET_FOTO} foto tercapai.")
                    break

    finally:
        cap.release()
        face_detection.close()
        cv2.destroyAllWindows()

    print(f"Total foto {nama}: {count_photos(folder)}")


def main() -> None:
    init_db()
    rows = list_siswa(KELAS_AKTIF)

    print("=" * 88)
    print("PEREKAMAN WAJAH SISWA BERDASARKAN DATA EXCEL")
    print("=" * 88)
    print(f"Kelas aktif: {KELAS_AKTIF}")

    if not rows:
        print("Belum ada siswa. Jalankan: python import_siswa_excel.py")
        return

    show_roster(rows)

    print("\nMode:")
    print("[1] Rekam urut dari siswa pertama yang belum mencapai target")
    print("[2] Rekam berdasarkan nomor absen")
    print("[3] Keluar")
    choice = input("Pilihan [default 1]: ").strip() or "1"

    if choice == "1":
        queue = [r for r in rows if count_photos(BASE_DIR / str(r["folder_wajah"])) < TARGET_FOTO]
        if not queue:
            print("Semua siswa sudah mencapai target foto.")
            return

        for idx, row in enumerate(queue, start=1):
            print(f"\n[{idx}/{len(queue)}] {int(row['no_absen']):02d} - {row['nis']} - {row['nama']}")
            raw = input("ENTER untuk mulai, q untuk berhenti: ").strip().lower()
            if raw == "q":
                break
            capture_student(row)

        print("\nPerekaman selesai. Jalankan: python build_face_embeddings.py")

    elif choice == "2":
        raw = input("Masukkan nomor absen: ").strip()
        if not raw.isdigit():
            print("Nomor absen harus angka.")
            return

        row = get_siswa_by_no_absen(int(raw), KELAS_AKTIF)
        if not row:
            print("Siswa tidak ditemukan.")
            return

        capture_student(row)
        print("\nJika data wajah berubah, jalankan ulang: python build_face_embeddings.py")

    else:
        print("Keluar.")


if __name__ == "__main__":
    main()
