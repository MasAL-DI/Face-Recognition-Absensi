"""
cek_skor_kamera.py
Alat bantu kalibrasi threshold LBPH.
Menampilkan prediksi NIS, nama, dan skor dari kamera tanpa mencatat absensi.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import mediapipe as mp

from config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, MATCH_THRESHOLD, MIN_DETECTION_CONFIDENCE, MODEL_DIR
from vision_utils import detection_to_bbox, draw_label, largest_detection, maybe_flip, preprocess_face_from_bbox, safe_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / MODEL_DIR / "face_model.yml"
LABEL_PATH = BASE_DIR / MODEL_DIR / "label_map.pkl"


def main() -> None:
    if not MODEL_PATH.exists() or not LABEL_PATH.exists():
        print("Model belum ada. Jalankan train_model.py.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    with open(LABEL_PATH, "rb") as f:
        label_map = pickle.load(f)

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

    print("Tekan ESC untuk keluar.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = maybe_flip(frame)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            display = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)

            if results.detections:
                det = largest_detection(results.detections, frame.shape)
                bbox = detection_to_bbox(det, frame.shape)
                face = preprocess_face_from_bbox(frame, bbox)
                if face is not None:
                    label_id, distance = recognizer.predict(face)
                    info = label_map.get(int(label_id), {})
                    nis = safe_text(info.get("nis", "-"))
                    nama = safe_text(info.get("nama", "-"))
                    no_absen = int(info.get("no_absen", 0))
                    color = (0, 220, 0) if distance <= MATCH_THRESHOLD else (0, 0, 220)
                    draw_label(display, bbox, f"{no_absen:02d} | {nis} | {nama} | skor {distance:.1f}", color)

            cv2.rectangle(display, (0, 0), (FRAME_WIDTH, 68), (24, 24, 36), -1)
            cv2.putText(display, f"CEK SKOR LBPH | threshold {MATCH_THRESHOLD} | ESC keluar", (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (255, 255, 255), 2)
            cv2.imshow("Cek Skor Kamera", display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        face_detection.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
