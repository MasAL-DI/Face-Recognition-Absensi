"""
Face recognition berbasis ArcFace embedding memakai InsightFace.

Alur:
1. build_face_embeddings.py membuat template embedding rata-rata per siswa.
2. absensi_lanjutan_arcface_cnn.py mengambil embedding wajah dari kamera.
3. Identitas dipilih berdasarkan cosine similarity tertinggi.

Nilai similarity makin besar = makin mirip.
Threshold awal di config_lanjutan.py wajib dikalibrasi dengan data kamera asli.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from config_lanjutan import ARCFACE_EMBEDDING_FILE, ARCFACE_MODEL_NAME, ARCFACE_SIMILARITY_THRESHOLD
except Exception:  # pragma: no cover
    ARCFACE_EMBEDDING_FILE = Path("model/arcface_embeddings.pkl")
    ARCFACE_MODEL_NAME = "buffalo_l"
    ARCFACE_SIMILARITY_THRESHOLD = 0.45


@dataclass
class FaceMatch:
    matched: bool
    info: Optional[dict]
    similarity: float
    bbox: Optional[tuple[int, int, int, int]]
    face_crop: Optional[np.ndarray]
    message: str


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


class ArcFaceEngine:
    def __init__(self, ctx_id: int = -1):
        self.ctx_id = ctx_id
        self.app = None
        self._load_app()

    def _load_app(self) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Library insightface belum terpasang. Jalankan: pip install insightface onnxruntime"
            ) from exc

        self.app = FaceAnalysis(name=ARCFACE_MODEL_NAME)
        self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))

    def get_largest_face(self, frame_bgr: np.ndarray):
        faces = self.app.get(frame_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    @staticmethod
    def face_bbox(face, frame_shape) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in face.bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        return x1, y1, x2, y2

    def extract_embedding(self, frame_bgr: np.ndarray) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]], Optional[np.ndarray]]:
        face = self.get_largest_face(frame_bgr)
        if face is None:
            return None, None, None
        emb = l2_normalize(np.asarray(face.embedding, dtype="float32"))
        bbox = self.face_bbox(face, frame_bgr.shape)
        x1, y1, x2, y2 = bbox
        crop = frame_bgr[y1:y2, x1:x2].copy()
        return emb, bbox, crop


class ArcFaceRecognizer:
    def __init__(
        self,
        embedding_file: Path | str = ARCFACE_EMBEDDING_FILE,
        threshold: float = ARCFACE_SIMILARITY_THRESHOLD,
        ctx_id: int = -1,
    ):
        self.embedding_file = Path(embedding_file)
        self.threshold = float(threshold)
        self.engine = ArcFaceEngine(ctx_id=ctx_id)
        self.templates: list[dict] = []
        self.load_templates()

    def load_templates(self) -> None:
        if not self.embedding_file.exists():
            raise FileNotFoundError(
                f"File embedding belum ada: {self.embedding_file}. Jalankan python build_face_embeddings.py"
            )
        with open(self.embedding_file, "rb") as f:
            payload = pickle.load(f)
        self.templates = payload.get("templates", [])
        if not self.templates:
            raise ValueError("Template embedding kosong. Rekam foto siswa dan jalankan build_face_embeddings.py")

    def identify(self, frame_bgr: np.ndarray) -> FaceMatch:
        emb, bbox, crop = self.engine.extract_embedding(frame_bgr)
        if emb is None:
            return FaceMatch(False, None, 0.0, None, None, "Wajah tidak terdeteksi")

        best_info = None
        best_sim = -1.0
        for item in self.templates:
            sim = cosine_similarity(emb, item["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_info = item

        if best_info is None:
            return FaceMatch(False, None, 0.0, bbox, crop, "Template siswa kosong")

        matched = best_sim >= self.threshold
        msg = f"Dikenali similarity {best_sim:.3f}" if matched else f"Tidak dikenal similarity {best_sim:.3f}"
        return FaceMatch(matched, best_info, best_sim, bbox, crop, msg)


def draw_face_label(frame, bbox, text, color=(0, 220, 0)):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    y_text = max(28, y1 - 10)
    cv2.rectangle(frame, (x1, y_text - 26), (min(frame.shape[1] - 1, x1 + 760), y_text + 4), color, -1)
    cv2.putText(frame, text[:82], (x1 + 6, y_text - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
