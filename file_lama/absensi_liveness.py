"""
absensi_liveness.py
Sistem absensi per sesi mata pelajaran dengan:
- face detection
- face recognition LBPH
- active liveness sederhana: blink challenge
- tampilan NIS + nama siswa
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

from config import (
    BLINK_TIMEOUT_SECONDS,
    CAMERA_INDEX,
    DETEKSI_BERULANG,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    KELAS_AKTIF,
    LIVENESS_METHOD,
    MATCH_THRESHOLD,
    MIN_DETECTION_CONFIDENCE,
    MODEL_DIR,
    USE_LIVENESS,
)
from database import (
    create_or_get_sesi,
    get_hadir_siswa_ids,
    init_db,
    list_siswa,
    record_absensi,
)
from vision_utils import (
    BlinkLiveness,
    detection_to_bbox,
    draw_label,
    largest_detection,
    maybe_flip,
    preprocess_face_from_bbox,
    safe_text,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / MODEL_DIR / "face_model.yml"
LABEL_PATH = BASE_DIR / MODEL_DIR / "label_map.pkl"


def input_with_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value if value else default


def prompt_nonempty(label: str) -> str:
    while True:
        value = input(label).strip()
        if value:
            return " ".join(value.split())
        print("Input tidak boleh kosong.")


def validate_hhmm(value: str) -> bool:
    try:
        datetime.strptime(value, "%H:%M")
        return True
    except ValueError:
        return False


def prompt_hhmm(label: str, default: str | None = None) -> str:
    while True:
        if default:
            value = input_with_default(label, default)
        else:
            value = input(label).strip()
        if validate_hhmm(value):
            return value
        print("Format jam harus HH:MM, contoh 07:30.")


def load_model():
    if not MODEL_PATH.exists() or not LABEL_PATH.exists():
        print("Model belum ada. Jalankan: python train_model.py")
        raise SystemExit

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    with open(LABEL_PATH, "rb") as f:
        label_map = pickle.load(f)

    return recognizer, label_map


def draw_right_panel(display, students, attended_ids):
    frame_h, frame_w, _ = display.shape
    panel_x = 780
    cv2.rectangle(display, (panel_x, 90), (frame_w, frame_h), (24, 24, 36), -1)

    hadir = [s for s in students if int(s["id_siswa"]) in attended_ids]
    belum = [s for s in students if int(s["id_siswa"]) not in attended_ids]

    cv2.putText(display, "HADIR", (panel_x + 16, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 255, 120), 2)
    y = 158
    for s in hadir[:15]:
        txt = f"{int(s['no_absen']):02d}. {safe_text(s['nis'])} | {safe_text(s['nama'])}"
        cv2.putText(display, txt, (panel_x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)
        y += 23

    if len(hadir) > 15:
        cv2.putText(display, f"... +{len(hadir) - 15}", (panel_x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)

    y_belum = max(428, y + 35)
    cv2.putText(display, "BELUM", (panel_x + 16, y_belum), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 170, 130), 2)
    y = y_belum + 32
    for s in belum[:10]:
        txt = f"{int(s['no_absen']):02d}. {safe_text(s['nis'])} | {safe_text(s['nama'])}"
        cv2.putText(display, txt, (panel_x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 190, 170), 1)
        y += 23

    if len(belum) > 10:
        cv2.putText(display, f"... +{len(belum) - 10}", (panel_x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 190, 170), 1)


def main() -> None:
    init_db()
    recognizer, label_map = load_model()

    students = list_siswa(KELAS_AKTIF)
    if not students:
        print("Belum ada siswa pada kelas aktif. Jalankan import_siswa_excel.py.")
        return
    students_by_id = {int(s["id_siswa"]): s for s in students}

    print("=" * 90)
    print("ABSENSI WAJAH + LIVENESS DETECTION")
    print("=" * 90)
    print(f"Kelas aktif: {KELAS_AKTIF}")
    print(f"Liveness   : {'aktif' if USE_LIVENESS else 'nonaktif'} ({LIVENESS_METHOD})")

    today = datetime.now().strftime("%Y-%m-%d")
    now_hhmm = datetime.now().strftime("%H:%M")

    tanggal = input_with_default("Tanggal", today)
    nama_mapel = prompt_nonempty("Mata pelajaran: ")
    nama_guru = input("Nama guru (opsional): ").strip()
    jam_mulai = prompt_hhmm("Jam mulai", now_hhmm)
    jam_selesai = prompt_hhmm("Jam selesai: ")

    sesi = create_or_get_sesi(
        tanggal=tanggal,
        kelas=KELAS_AKTIF,
        nama_mapel=nama_mapel,
        jam_mulai=jam_mulai,
        jam_selesai=jam_selesai,
        nama_guru=nama_guru,
    )

    attended_ids = get_hadir_siswa_ids(int(sesi["id_sesi"]))

    print("\nSesi aktif:")
    print(f"ID sesi       : {sesi['id_sesi']}")
    print(f"Tanggal       : {sesi['tanggal']}")
    print(f"Kelas         : {sesi['kelas']}")
    print(f"Mata pelajaran: {sesi['nama_mapel']}")
    print(f"Jam           : {sesi['jam_mulai']} - {sesi['jam_selesai']}")
    print(f"Sudah hadir   : {len(attended_ids)}/{len(students)}")
    print(f"Threshold     : skor <= {MATCH_THRESHOLD}")
    print("\nTekan ESC untuk selesai.\n")

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka.")
        face_detection.close()
        face_mesh.close()
        return

    current_id: int | None = None
    stable_count = 0
    last_score: float | None = None
    liveness = BlinkLiveness()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca kamera.")
                break

            frame = maybe_flip(frame)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            display = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_results = face_detection.process(rgb)
            mesh_results = face_mesh.process(rgb)

            face_landmarks = None
            if mesh_results.multi_face_landmarks:
                face_landmarks = mesh_results.multi_face_landmarks[0]

            recognized_id = None
            recognized_info = None
            recognized_distance = None
            bbox = None
            status_text = "Arahkan satu wajah ke kamera"

            if detection_results.detections:
                det = largest_detection(detection_results.detections, frame.shape)

                if det and float(det.score[0]) >= MIN_DETECTION_CONFIDENCE:
                    bbox = detection_to_bbox(det, frame.shape)
                    face = preprocess_face_from_bbox(frame, bbox)

                    if face is not None:
                        label_id, distance = recognizer.predict(face)
                        recognized_distance = float(distance)
                        info = label_map.get(int(label_id))

                        if info and distance <= MATCH_THRESHOLD:
                            candidate_id = int(info["id_siswa"])

                            if candidate_id in students_by_id:
                                recognized_id = candidate_id
                                recognized_info = info

                                if current_id != recognized_id:
                                    current_id = recognized_id
                                    stable_count = 0
                                    liveness.reset(recognized_id)

                                stable_count += 1
                                last_score = float(distance)

                                no_absen = int(info.get("no_absen", 0))
                                nis = safe_text(info.get("nis", "-"))
                                nama = safe_text(info.get("nama", "-"))
                                label_siswa = f"{no_absen:02d} | {nis} | {nama}"

                                if recognized_id in attended_ids:
                                    draw_label(display, bbox, f"{label_siswa} | sudah hadir | skor {distance:.1f}", (0, 165, 255))
                                    status_text = "Siswa sudah tercatat hadir pada sesi ini."
                                else:
                                    live_ok = True
                                    live_instruction = "Liveness nonaktif"
                                    if USE_LIVENESS:
                                        live_ok = liveness.update(recognized_id, face_landmarks)
                                        live_instruction = liveness.instruction()
                                        if liveness.expired() and not live_ok:
                                            liveness.reset(recognized_id)

                                    if live_ok:
                                        draw_label(display, bbox, f"{label_siswa} | liveness OK | skor {distance:.1f}", (0, 220, 0))
                                    else:
                                        draw_label(display, bbox, f"{label_siswa} | {live_instruction} | skor {distance:.1f}", (0, 200, 255))

                                    status_text = f"{label_siswa} | stabil {stable_count}/{DETEKSI_BERULANG} | {live_instruction}"

                                    if stable_count >= DETEKSI_BERULANG and live_ok:
                                        jam_absen = datetime.now().strftime("%H:%M:%S")
                                        inserted = record_absensi(
                                            id_sesi=int(sesi["id_sesi"]),
                                            id_siswa=recognized_id,
                                            jam_absen=jam_absen,
                                            skor_lbph=last_score,
                                            liveness_method=LIVENESS_METHOD if USE_LIVENESS else "OFF",
                                            liveness_passed=bool(live_ok),
                                            liveness_detail=f"EAR terakhir: {liveness.last_ear:.3f}" if USE_LIVENESS else "Liveness OFF",
                                        )

                                        if inserted:
                                            attended_ids.add(recognized_id)
                                            print(f"HADIR: {label_siswa} pukul {jam_absen}, skor {last_score:.1f}")
                                        stable_count = 0
                                        liveness.reset(recognized_id)

                            else:
                                draw_label(display, bbox, "Wajah dikenali, tetapi bukan kelas aktif", (160, 80, 255))
                        else:
                            if bbox is not None:
                                draw_label(display, bbox, f"Tidak dikenal | skor {distance:.1f}", (0, 0, 220))
                            current_id = None
                            stable_count = 0
                            liveness.reset(None)

            # Header
            cv2.rectangle(display, (0, 0), (FRAME_WIDTH, 90), (24, 24, 36), -1)
            cv2.putText(
                display,
                f"ABSENSI + LIVENESS | {KELAS_AKTIF} | {sesi['nama_mapel']}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                f"{sesi['tanggal']} | {sesi['jam_mulai']}-{sesi['jam_selesai']} | Hadir {len(attended_ids)}/{len(students)} | Threshold {MATCH_THRESHOLD} | ESC keluar",
                (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (190, 220, 255),
                2,
            )

            # Status bawah
            cv2.rectangle(display, (0, FRAME_HEIGHT - 54), (FRAME_WIDTH, FRAME_HEIGHT), (24, 24, 36), -1)
            cv2.putText(
                display,
                status_text[:110],
                (16, FRAME_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (220, 240, 255),
                2,
            )

            if USE_LIVENESS and current_id is not None and current_id not in attended_ids:
                detail = f"EAR: {liveness.last_ear:.3f} | Timeout: {BLINK_TIMEOUT_SECONDS}s"
                cv2.putText(
                    display,
                    detail,
                    (16, FRAME_HEIGHT - 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (200, 255, 200),
                    1,
                )

            draw_right_panel(display, students, attended_ids)

            cv2.imshow("Absensi Wajah dengan Liveness Detection", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        cap.release()
        face_detection.close()
        face_mesh.close()
        cv2.destroyAllWindows()

    print("\n" + "=" * 90)
    print("SESI SELESAI")
    print(f"Sesi  : {KELAS_AKTIF} | {sesi['nama_mapel']} | {sesi['tanggal']}")
    print(f"Hadir : {len(attended_ids)}/{len(students)} siswa")
    print("Laporan: python rekap.py")
    print("=" * 90)


if __name__ == "__main__":
    main()
