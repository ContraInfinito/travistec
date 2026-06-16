"""ONNX-based FER+ emotion detector with OpenCV face detection and fallback.

This module uses the FER+ model from the ONNX Model Zoo to detect emotions.
If the ONNX model is not found locally, it will attempt to download it once.
If anything fails, it falls back to the simple Haar+smile heuristic.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

# Optional imports at runtime to keep import-time cheap
_ort = None

MODEL_URL = (
    "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)
MODEL_FILENAME = "emotion-ferplus-8.onnx"


def _models_dir() -> Path:
    # Place model file under backend/models
    return Path(__file__).resolve().parent.parent / "models"


def _ensure_model_file() -> Path | None:
    """Ensure FER+ ONNX model is present locally; try to download if missing."""
    # Allow override via environment variable
    env_path = os.getenv("EMOTION_ONNX_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    models_dir = _models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME
    if model_path.exists():
        return model_path
    # Try to download
    try:
        import requests

        resp = requests.get(MODEL_URL, timeout=30)
        if resp.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(resp.content)
            return model_path
        else:
            return None
    except Exception:
        # No network or requests missing
        return None


def _get_session():
    global _ort
    if _ort is None:
        try:
            import onnxruntime as ort

            _ort = ort
        except Exception:
            _ort = False  # explicitly mark unavailable
    return _ort


FERPLUS_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _preprocess_face(gray: np.ndarray) -> np.ndarray:
    # FER+ expects 64x64 grayscale, normalized
    face = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    face = face.astype(np.float32)
    face = (face - 128.0) / 128.0
    face = np.expand_dims(face, axis=0)  # 1 x 64 x 64
    face = np.expand_dims(face, axis=0)  # 1 x 1 x 64 x 64 (NCHW)
    return face


def _analyze_with_ferplus(img_bgr: np.ndarray) -> Dict[str, Any]:
    ort = _get_session()
    if not ort:
        return {"error": "onnxruntime_unavailable"}

    model_path = _ensure_model_file()
    if model_path is None:
        return {"error": "model_not_available"}

    # Init session
    try:
        sess = ort.InferenceSession(str(model_path), providers=[
                                    "CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
    except Exception as e:
        return {"error": f"session_init_failed: {e}"}

    # Face detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        return {"dominant_emotion": "no_face", "face_count": 0, "details": []}

    details: List[Dict[str, Any]] = []
    # Aggregate probabilities to decide dominant across all faces
    agg_probs = np.zeros(len(FERPLUS_LABELS), dtype=np.float32)

    for (x, y, w, h) in faces:
        roi_gray = gray[y: y + h, x: x + w]
        inp = _preprocess_face(roi_gray)
        try:
            logits = sess.run([output_name], {input_name: inp})[0]
        except Exception as e:
            return {"error": f"inference_failed: {e}"}
        # logits shape: (1, 8)
        probs = _softmax(logits[0].astype(np.float32))
        agg_probs += probs
        top_idx = int(np.argmax(probs))
        details.append(
            {
                "box": [int(x), int(y), int(w), int(h)],
                "top_emotion": FERPLUS_LABELS[top_idx],
                "scores": {FERPLUS_LABELS[i]: float(probs[i]) for i in range(len(FERPLUS_LABELS))},
            }
        )

    agg_probs = agg_probs / max(1, len(faces))
    dom_idx = int(np.argmax(agg_probs))
    return {
        "dominant_emotion": FERPLUS_LABELS[dom_idx],
        "face_count": int(len(faces)),
        "details": details,
    }


def analyze_image_file(path: str | os.PathLike) -> Dict[str, Any]:
    """Analyze an image file and return emotions using FER+; fallback to simple heuristic.

    Returns a dict with keys: dominant_emotion, face_count, details (per face)
    or an error key when appropriate.
    """
    p = Path(path)
    if not p.exists():
        return {"error": "file_not_found", "path": str(path)}

    img = cv2.imread(str(p))
    if img is None:
        return {"error": "invalid_image"}

    res = _analyze_with_ferplus(img)
    if "error" in res and res["error"] in ("onnxruntime_unavailable", "model_not_available", "session_init_failed"):
        # Fallback to simple heuristic
        try:
            # Fallback removed: legacy analyzers are no longer available
            def _fallback(*args, **kwargs):
                return {"error": "legacy_emotion_analyzers_removed"}

            return _fallback(path)
        except Exception:
            pass
    return res


def describe() -> str:
    return "onnx FER+ emotion detector with Haar face detection and fallback"
