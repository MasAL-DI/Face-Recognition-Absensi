"""
Absensi lanjutan:
- Face recognition berbasis ArcFace embedding
- Anti-spoofing foto/video berbasis CNN
- Blink challenge sebagai liveness aktif tambahan
- Tetap memakai database.py lama untuk sesi dan pencatatan absensi

Jalankan setelah:
1. python import_siswa_excel.py
2. python daftar_wajah.py
3. python build_face_embeddings.py
4. python train_spoof_cnn.py  # bila ingin anti-spoof CNN aktif
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

from anti_spoof_cnn import CNNAntiSpoofDetector
from arcface_recognition import ArcFaceRecognizer, draw_face_label
from config_lanjutan import (
    ARCFACE_SIMILARITY_THRESHOLD,
    BLINK_TIMEOUT_SECONDS,
    CAMERA_INDEX,
    DETEKSI_BERULANG,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    KELAS_AKTIF,
    LIVENESS_METHOD,
    MIN_DETECTION_CONFIDENCE,
    USE_BLINK_CHALLENGE,
    USE_CNN_ANTI_SPOOF,
)
from database import (
    create_or_get_sesi,
    get_hadir_siswa_ids,
    init_db,
    list_siswa,
    record_absensi,
)
from vision_utils import BlinkLiveness, maybe_flip, safe_text

BASE_DIR = Path(__file__).resolve().parent

# ArcFace cukup berat di CPU. Jalankan identifikasi tiap beberapa frame agar FaceMesh tetap sempat menangkap kedipan.
ARCFACE_EVERY_N_FRAMES = 3


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
        value = input_with_default(label, default) if default else input(label).strip()
        if validate_hhmm(value):
            return value
        print("Format jam harus HH:MM, contoh 07:30.")


def draw_header(display, sesi, attended_count: int, total_students: int) -> None:
    cv2.rectangle(display, (0, 0), (FRAME_WIDTH, 92), (18, 24, 38), -1)
    cv2.putText(
        display,
        f"ABSENSI LANJUTAN | ARCFACE + CNN ANTI-SPOOF | {KELAS_AKTIF}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        display,
        f"{sesi['tanggal']} | {sesi['nama_mapel']} | {sesi['jam_mulai']}-{sesi['jam_selesai']} | Hadir {attended_count}/{total_students} | Threshold ArcFace {ARCFACE_SIMILARITY_THRESHOLD}",
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (205, 225, 255),
        1,
    )


def draw_bottom_status(display, text: str) -> None:
    cv2.rectangle(display, (0, FRAME_HEIGHT - 60), (FRAME_WIDTH, FRAME_HEIGHT), (18, 24, 38), -1)
    cv2.putText(
        display,
        text[:120],
        (16, FRAME_HEIGHT - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (225, 240, 255),
        2,
    )


def draw_side_panel(display, students, attended_ids) -> None:
    frame_h, frame_w = display.shape[:2]
    panel_w = min(280, max(220, frame_w // 3))
    panel_x = max(0, frame_w - panel_w)
    cv2.rectangle(display, (panel_x, 92), (frame_w, frame_h - 60), (18, 24, 38), -1)

    hadir = [s for s in students if int(s["id_siswa"]) in attended_ids]
    belum = [s for s in students if int(s["id_siswa"]) not in attended_ids]

    cv2.putText(display, "HADIR", (panel_x + 16, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (120, 255, 120), 2)
    y = 164
    for s in hadir[:15]:
        txt = f"{int(s['no_absen']):02d}. {safe_text(s['nis'])} | {safe_text(s['nama'])}"
        cv2.putText(display, txt[:45], (panel_x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 255, 190), 1)
        y += 23

    if len(hadir) > 15:
        cv2.putText(display, f"... +{len(hadir) - 15}", (panel_x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 255, 190), 1)

    y_belum = max(430, y + 35)
    cv2.putText(display, "BELUM", (panel_x + 16, y_belum), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 175, 130), 2)
    y = y_belum + 32
    for s in belum[:10]:
        txt = f"{int(s['no_absen']):02d}. {safe_text(s['nis'])} | {safe_text(s['nama'])}"
        cv2.putText(display, txt[:45], (panel_x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 195, 170), 1)
        y += 23

    if len(belum) > 10:
        cv2.putText(display, f"... +{len(belum) - 10}", (panel_x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 195, 170), 1)


def main() -> None:
    init_db()

    students = list_siswa(KELAS_AKTIF)
    if not students:
        print("Belum ada siswa pada kelas aktif. Jalankan import_siswa_excel.py.")
        return
    students_by_id = {int(s["id_siswa"]): s for s in students}

    print("=" * 90)
    print("ABSENSI LANJUTAN: ARCFACE + CNN ANTI-SPOOF + BLINK")
    print("=" * 90)
    print(f"Kelas aktif : {KELAS_AKTIF}")
    print(f"Anti-spoof  : {'aktif' if USE_CNN_ANTI_SPOOF else 'nonaktif'}")
    print(f"Blink       : {'aktif' if USE_BLINK_CHALLENGE else 'nonaktif'}")

    recognizer = ArcFaceRecognizer(threshold=ARCFACE_SIMILARITY_THRESHOLD, ctx_id=-1)
    spoof_detector = CNNAntiSpoofDetector()

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
        face_mesh.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    current_id: int | None = None
    stable_count = 0
    liveness = BlinkLiveness()
    last_similarity = 0.0
    last_spoof_detail = "-"
    frame_count = 0
    last_match = None

    print("Tekan ESC untuk keluar.")

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
            mesh_results = face_mesh.process(rgb)
            face_landmarks = mesh_results.multi_face_landmarks[0] if mesh_results.multi_face_landmarks else None

            frame_count += 1
            if last_match is None or frame_count % ARCFACE_EVERY_N_FRAMES == 0:
                last_match = recognizer.identify(frame)
            match = last_match
            status_text = match.message

            if match.bbox is not None:
                if not match.matched or not match.info:
                    draw_face_label(display, match.bbox, match.message, (0, 0, 220))
                    current_id = None
                    stable_count = 0
                    liveness.reset(None)
                else:
                    info = match.info
                    candidate_id = int(info["id_siswa"])
                    last_similarity = float(match.similarity)

                    if candidate_id not in students_by_id:
                        draw_face_label(display, match.bbox, "Wajah dikenali, tetapi bukan kelas aktif", (160, 80, 255))
                        current_id = None
                        stable_count = 0
                        liveness.reset(None)
                    else:
                        if current_id != candidate_id:
                            current_id = candidate_id
                            stable_count = 0
                            liveness.reset(candidate_id)

                        no_absen = int(info.get("no_absen", 0))
                        nis = safe_text(info.get("nis", "-"))
                        nama = safe_text(info.get("nama", "-"))
                        label_siswa = f"{no_absen:02d} | {nis} | {nama}"

                        if candidate_id in attended_ids:
                            draw_face_label(display, match.bbox, f"{label_siswa} | sudah hadir | sim {last_similarity:.3f}", (0, 165, 255))
                            status_text = "Siswa sudah tercatat hadir pada sesi ini."
                            stable_count = 0
                        else:
                            spoof_ok = True
                            spoof_detail = "Anti-spoof nonaktif"
                            if USE_CNN_ANTI_SPOOF:
                                spoof_result = spoof_detector.predict(match.face_crop)
                                spoof_ok = spoof_result.is_real
                                spoof_detail = spoof_result.status
                                last_spoof_detail = spoof_detail

                            blink_ok = True
                            blink_instruction = "Blink nonaktif"
                            if USE_BLINK_CHALLENGE:
                                blink_ok = liveness.update(candidate_id, face_landmarks)
                                blink_instruction = liveness.instruction()
                                if liveness.expired() and not blink_ok:
                                    liveness.reset(candidate_id)

                            if spoof_ok and blink_ok:
                                stable_count += 1
                                draw_face_label(
                                    display,
                                    match.bbox,
                                    f"{label_siswa} | REAL + LIVE | sim {last_similarity:.3f}",
                                    (0, 220, 0),
                                )
                            else:
                                stable_count = 0
                                draw_face_label(
                                    display,
                                    match.bbox,
                                    f"{label_siswa} | {spoof_detail} | {blink_instruction}",
                                    (0, 200, 255) if spoof_ok else (0, 0, 220),
                                )

                            status_text = (
                                f"{label_siswa} | stabil {stable_count}/{DETEKSI_BERULANG} | "
                                f"CNN: {spoof_detail} | Blink: {blink_instruction}"
                            )

                            if stable_count >= DETEKSI_BERULANG and spoof_ok and blink_ok:
                                jam_absen = datetime.now().strftime("%H:%M:%S")
                                inserted = record_absensi(
                                    id_sesi=int(sesi["id_sesi"]),
                                    id_siswa=candidate_id,
                                    jam_absen=jam_absen,
                                    skor_lbph=last_similarity,
                                    liveness_method=LIVENESS_METHOD,
                                    liveness_passed=True,
                                    liveness_detail=(
                                        f"ArcFace similarity={last_similarity:.3f}; "
                                        f"CNN={last_spoof_detail}; EAR={liveness.last_ear:.3f}"
                                    ),
                                )
                                if inserted:
                                    attended_ids.add(candidate_id)
                                    print(f"HADIR: {label_siswa} pukul {jam_absen}, sim {last_similarity:.3f}, {last_spoof_detail}")
                                stable_count = 0
                                liveness.reset(candidate_id)

            draw_header(display, sesi, len(attended_ids), len(students))
            draw_side_panel(display, students, attended_ids)
            if USE_BLINK_CHALLENGE and current_id is not None and current_id not in attended_ids:
                cv2.putText(
                    display,
                    f"EAR: {liveness.last_ear:.3f} | Timeout: {BLINK_TIMEOUT_SECONDS}s",
                    (16, FRAME_HEIGHT - 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (200, 255, 200),
                    1,
                )
            draw_bottom_status(display, status_text)

            cv2.imshow("Absensi Lanjutan - ArcFace + CNN Anti-Spoof", display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()

    print("=" * 90)
    print("SESI SELESAI")
    print(f"Sesi  : {KELAS_AKTIF} | {sesi['nama_mapel']} | {sesi['tanggal']}")
    print(f"Hadir : {len(attended_ids)}/{len(students)} siswa")
    print("Dashboard: streamlit run web_dashboard.py")
    print("=" * 90)


if __name__ == "__main__":
    main()
