"""
vision_utils.py
Fungsi deteksi, preprocessing wajah, dan liveness blink challenge.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2

try:
    # Untuk versi lanjutan, pakai konfigurasi dari config_lanjutan agar tuning blink tidak diabaikan.
    from config_lanjutan import (
        BLINK_EAR_THRESHOLD,
        BLINK_MIN_CLOSED_FRAMES,
        BLINK_TIMEOUT_SECONDS,
        FLIP_HORIZONTAL,
        MIN_DETECTION_CONFIDENCE,
    )
except Exception:  # fallback untuk versi lama/LBPH
    from config import (
        BLINK_EAR_THRESHOLD,
        BLINK_MIN_CLOSED_FRAMES,
        BLINK_TIMEOUT_SECONDS,
        FLIP_HORIZONTAL,
        MIN_DETECTION_CONFIDENCE,
    )

FACE_SIZE = (200, 200)
BBOX_MARGIN_RATIO = 0.18


def maybe_flip(frame):
    return cv2.flip(frame, 1) if FLIP_HORIZONTAL else frame


def expand_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_w: int,
    frame_h: int,
    margin_ratio: float = BBOX_MARGIN_RATIO,
) -> tuple[int, int, int, int]:
    box_w = x2 - x1
    box_h = y2 - y1

    margin_x = int(box_w * margin_ratio)
    margin_y = int(box_h * margin_ratio)

    nx1 = max(0, x1 - margin_x)
    ny1 = max(0, y1 - margin_y)
    nx2 = min(frame_w, x2 + margin_x)
    ny2 = min(frame_h, y2 + margin_y)
    return nx1, ny1, nx2, ny2


def preprocess_face_from_bbox(frame, bbox: tuple[int, int, int, int]):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.resize(gray, FACE_SIZE)


def prepare_face_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, FACE_SIZE)
    return cv2.equalizeHist(img)


def draw_label(frame, bbox: tuple[int, int, int, int], text: str, color: tuple[int, int, int]):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.56
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    label_y1 = y1 - th - 16
    label_y2 = y1 - 6
    if label_y1 < 0:
        label_y1 = y2 + 6
        label_y2 = y2 + th + 16

    label_x2 = min(frame.shape[1] - 1, x1 + tw + 12)
    cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 6, label_y2 - 6),
        font,
        scale,
        (255, 255, 255),
        thickness,
    )


def safe_text(text, fallback: str = "-") -> str:
    cleaned = " ".join(str(text or "").strip().split())
    return cleaned if cleaned else fallback


def largest_detection(detections, frame_shape):
    if not detections:
        return None

    h, w, _ = frame_shape

    def area(det):
        bb = det.location_data.relative_bounding_box
        return max(0, bb.width * w) * max(0, bb.height * h)

    return max(detections, key=area)


def detection_to_bbox(det, frame_shape) -> tuple[int, int, int, int]:
    h, w, _ = frame_shape
    bb = det.location_data.relative_bounding_box
    x1 = max(0, int(bb.xmin * w))
    y1 = max(0, int(bb.ymin * h))
    x2 = min(w, int((bb.xmin + bb.width) * w))
    y2 = min(h, int((bb.ymin + bb.height) * h))
    return expand_bbox(x1, y1, x2, y2, w, h)


def euclidean(p1, p2) -> float:
    return math.dist((p1.x, p1.y), (p2.x, p2.y))


# MediaPipe FaceMesh landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def eye_aspect_ratio(landmarks, eye_indices: list[int]) -> float:
    p = [landmarks[i] for i in eye_indices]
    horizontal = euclidean(p[0], p[3])
    if horizontal == 0:
        return 1.0

    vertical_1 = euclidean(p[1], p[5])
    vertical_2 = euclidean(p[2], p[4])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_ear(face_landmarks) -> float:
    landmarks = face_landmarks.landmark
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
    return (left_ear + right_ear) / 2.0


@dataclass
class BlinkLiveness:
    """
    Active liveness sederhana:
    - state awal menunggu wajah dikenali
    - siswa diminta berkedip
    - pass jika mata tertutup beberapa frame lalu terbuka kembali
    """
    subject_id: int | None = None
    start_time: float = 0.0
    closed_frames: int = 0
    saw_closed: bool = False
    passed: bool = False
    last_ear: float = 0.0

    def reset(self, subject_id: int | None = None):
        self.subject_id = subject_id
        self.start_time = time.time()
        self.closed_frames = 0
        self.saw_closed = False
        self.passed = False
        self.last_ear = 0.0

    def expired(self) -> bool:
        if self.start_time <= 0:
            return False
        return (time.time() - self.start_time) > BLINK_TIMEOUT_SECONDS

    def update(self, subject_id: int, face_landmarks) -> bool:
        if self.subject_id != subject_id:
            self.reset(subject_id)

        if face_landmarks is None:
            return self.passed

        ear = calculate_ear(face_landmarks)
        self.last_ear = ear

        if ear < BLINK_EAR_THRESHOLD:
            self.closed_frames += 1
            if self.closed_frames >= BLINK_MIN_CLOSED_FRAMES:
                self.saw_closed = True
        else:
            # Mata terbuka lagi setelah sempat tertutup = blink valid.
            if self.saw_closed:
                self.passed = True
            self.closed_frames = 0

        return self.passed

    def instruction(self) -> str:
        if self.passed:
            return "Liveness OK"
        if self.expired():
            return "Ulangi kedipan"
        return "KEDIPKAN MATA"
