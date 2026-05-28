"""
Inference CNN anti-spoofing untuk membedakan wajah asli vs spoof foto/video.

Cara kerja:
- menerima crop wajah dari frame kamera
- resize ke input CNN
- model menghasilkan skor spoof
- real_score = 1 - spoof_score

Model harus dilatih dulu dengan train_spoof_cnn.py.
Dataset minimum yang disarankan:
- dataset_spoof/train/real  : wajah asli dari kamera
- dataset_spoof/train/spoof : foto HP, foto cetak, video replay
- dataset_spoof/val/real
- dataset_spoof/val/spoof
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from config_lanjutan import (
        ALLOW_ABSEN_IF_SPOOF_MODEL_MISSING,
        SPOOF_INPUT_SIZE,
        SPOOF_MODEL_FILE,
        SPOOF_REAL_THRESHOLD,
    )
except Exception:  # pragma: no cover
    ALLOW_ABSEN_IF_SPOOF_MODEL_MISSING = False
    SPOOF_INPUT_SIZE = 224
    SPOOF_MODEL_FILE = Path("model/spoof_cnn.keras")
    SPOOF_REAL_THRESHOLD = 0.70


@dataclass
class SpoofResult:
    is_real: bool
    real_score: float
    spoof_score: float
    status: str


class CNNAntiSpoofDetector:
    def __init__(self, model_path: Path | str = SPOOF_MODEL_FILE, threshold: float = SPOOF_REAL_THRESHOLD):
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self.model = None
        self.ready = False
        self.error_message = ""
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self.error_message = f"Model anti-spoof belum ada: {self.model_path}"
            return
        try:
            from tensorflow.keras.models import load_model

            self.model = load_model(self.model_path)
            self.ready = True
        except Exception as exc:  # pragma: no cover
            self.error_message = f"Gagal memuat model anti-spoof: {exc}"
            self.ready = False

    @staticmethod
    def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Crop wajah kosong.")
        resized = cv2.resize(face_bgr, (SPOOF_INPUT_SIZE, SPOOF_INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype("float32") / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, face_bgr: Optional[np.ndarray]) -> SpoofResult:
        if face_bgr is None or face_bgr.size == 0:
            return SpoofResult(False, 0.0, 1.0, "Crop wajah kosong")

        if not self.ready:
            if ALLOW_ABSEN_IF_SPOOF_MODEL_MISSING:
                return SpoofResult(True, 1.0, 0.0, "Model anti-spoof belum ada, dilewati")
            return SpoofResult(False, 0.0, 1.0, self.error_message or "Model anti-spoof belum siap")

        try:
            x = self._preprocess(face_bgr)
            pred = float(self.model.predict(x, verbose=0).ravel()[0])

            # Pada train_spoof_cnn.py, label binary: 0 = real, 1 = spoof.
            spoof_score = max(0.0, min(1.0, pred))
            real_score = 1.0 - spoof_score
            is_real = real_score >= self.threshold
            status = f"REAL {real_score:.2f}" if is_real else f"SPOOF {spoof_score:.2f}"
            return SpoofResult(is_real, real_score, spoof_score, status)
        except Exception as exc:  # pragma: no cover
            return SpoofResult(False, 0.0, 1.0, f"Error anti-spoof: {exc}")
