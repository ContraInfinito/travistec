"""DeepFace-based emotion detector service.

Provides analyze_image_file(path) -> dict with keys:

If DeepFace fails to detect, falls back to the simple Haar+smile heuristic.
"""
from __future__ import annotations
import cv2

import os
from pathlib import Path
from typing import Any, Dict, List

# Improve DeepFace compatibility with TF / Keras 3 environments
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
# Quiet TensorFlow C++ logs (optional)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Ensure DeepFace uses legacy Keras API with TF >= 2.20
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def _analyze_with_deepface(img_path: str) -> Dict[str, Any]:
    from deepface import DeepFace

    detector_backend = os.getenv("EMOTION_DETECTOR_BACKEND", "opencv")
    # Analyze only emotions; allow no-face images to pass (we'll handle)
    res = DeepFace.analyze(
        img_path=img_path,
        actions=["emotion"],
        detector_backend=detector_backend,
        enforce_detection=False,
    )

    # DeepFace returns a dict or a list of dicts depending on faces
    results: List[Dict[str, Any]]
    if isinstance(res, list):
        results = res
    else:
        results = [res]

    details: List[Dict[str, Any]] = []
    for r in results:
        # region can be None if detection disabled or failed
        region = r.get("region") or {}
        box = [int(region.get(k, 0)) for k in ("x", "y", "w", "h")]
        de = r.get("dominant_emotion") or "neutral"
        scores = r.get("emotion") or {}
        # normalize scores to floats
        scores = {str(k): float(v) for k, v in scores.items()}
        details.append({
            "box": box,
            "top_emotion": de,
            "scores": scores,
        })

    # Aggregate dominant across faces by averaging probabilities if available
    agg: Dict[str, float] = {}
    for d in details:
        for k, v in d["scores"].items():
            agg[k] = agg.get(k, 0.0) + v
    if details:
        # choose max mean
        n = float(len(details))
        dominant = max(
            agg.items(), key=lambda kv: kv[1] / n)[0] if agg else details[0]["top_emotion"]
    else:
        dominant = "no_face"

    return {
        "dominant_emotion": dominant,
        "face_count": len(details),
        "details": details,
    }


def analyze_image_file(path: str | os.PathLike) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"error": "file_not_found", "path": str(path)}

    # DeepFace can read .webp, but we can also validate OpenCV load
    img = cv2.imread(str(p))
    if img is None:
        return {"error": "invalid_image"}

    try:
        return _analyze_with_deepface(str(p))
    except Exception as e:
        # No fallbacks per request; return explicit error so frontend can inform user
        return {"error": "deepface_unavailable", "detail": str(e)}
